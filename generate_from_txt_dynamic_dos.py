import argparse
import re
from pathlib import Path

import torch

# Compatibility shim: older diffusers versions import hf_cache_home from huggingface_hub.constants.
# Newer huggingface_hub releases removed it in favor of HF_HOME.
import huggingface_hub
import huggingface_hub.constants as hf_constants

if not hasattr(hf_constants, "hf_cache_home"):
    hf_constants.hf_cache_home = hf_constants.HF_HOME

# Compatibility shim: older diffusers versions import these names from huggingface_hub.
if not hasattr(huggingface_hub, "cached_download"):
    def _cached_download_compat(*args, **kwargs):
        return huggingface_hub.hf_hub_download(*args, **kwargs)

    huggingface_hub.cached_download = _cached_download_compat

if not hasattr(huggingface_hub, "HfFolder"):
    class _HfFolderCompat:
        @staticmethod
        def path_token():
            return hf_constants.HF_TOKEN_PATH

        @staticmethod
        def get_token():
            return huggingface_hub.get_token()

    huggingface_hub.HfFolder = _HfFolderCompat

from diffusers import StableDiffusionXLPipeline

import configs.envs  # noqa: F401 - keeps env side effects (e.g., HF token setup)
from methods.dynamic_dos import DynamicDOS, SDXLTextEncoder


OBJECT_PATTERN = re.compile(
    r"\b(?:a|an|the)\s+([a-zA-Z][a-zA-Z0-9\- ]*?)(?=\s+(?:and|with|in|on|at|under|over|beside|near|behind|inside|outside|while)\b|[,.;!?]|$)",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images for prompts in a text file using SDXL and DynamicDOS-SDXL."
    )
    parser.add_argument("--prompts_txt", default="/home/saket_p/multi_object_generation/T2I-CompBench/examples/dataset/3d_spatial_train.txt", help="Path to text file containing prompts.")
    parser.add_argument("--output_path", default="/home/saket_p/multi_object_generation/testing_output", help="Root output directory.")
    parser.add_argument("--device", default="cuda:0", help="Device to run on, e.g. cuda:0 or mps.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed. Effective seed is seed + prompt_index.")
    parser.add_argument(
        "--num_prompts",
        type=int,
        default=1,
        help="Maximum number of prompts to process. Uses min(num_prompts, prompts_in_file).",
    )
    parser.add_argument("--lambda_sep", type=float, default=1.0, help="DOS lambda_sep value.")
    parser.add_argument("--disable_separating_object_embedding", action="store_true")
    parser.add_argument("--disable_separating_eot_embedding", action="store_true")
    parser.add_argument("--disable_separating_pooled_embedding", action="store_true")
    parser.add_argument("--schedule_type", type=str, default="cosine", help="interpolation function for alpha")
    parser.add_argument("--tau", type=float, default=None, help="tau value for schedule_type")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--base_subdir", default="sdxl", help="Subdirectory name for baseline SDXL outputs.")
    parser.add_argument("--dydos_subdir", default="dydos_sdxl", help="Subdirectory name for DyDOS-SDXL outputs.")
    return parser.parse_args()


def _normalize_objects(raw_objects):
    cleaned = []
    seen = set()
    for item in raw_objects:
        value = item.strip().strip("\"'")
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(value)
    return cleaned


def infer_objects_from_prompt(prompt):
    matches = [m.group(1).strip() for m in OBJECT_PATTERN.finditer(prompt)]
    return _normalize_objects(matches)


def parse_prompt_line(line):
    # Supported explicit-object formats:
    # 1) prompt || object1, object2
    # 2) prompt<TAB>object1,object2
    if "||" in line:
        prompt, objects_raw = line.split("||", 1)
        objects = _normalize_objects(objects_raw.split(","))
        return prompt.strip(), objects

    if "\t" in line:
        prompt, objects_raw = line.split("\t", 1)
        objects = _normalize_objects(objects_raw.split(","))
        return prompt.strip(), objects

    prompt = line.strip()
    objects = infer_objects_from_prompt(prompt)
    return prompt, objects


def read_prompt_entries(prompts_txt):
    entries = []
    with open(prompts_txt, "r", encoding="utf-8") as f:
        for line_number, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue

            prompt, objects = parse_prompt_line(line)
            if not prompt:
                print(f"[WARN] Empty prompt at line {line_number}, skipping.")
                continue
            entries.append((prompt, objects, line_number))
    return entries


def load_sdxl_pipeline(device):
    return StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    ).to(device)


def generate_sdxl(pipe, prompt, seed, device):
    generator = torch.Generator(device).manual_seed(seed)
    return pipe(prompt=[prompt], generator=generator).images[0]


def generate_dos_sdxl(pipe, prompt, objects, seed, args):
    text_embeddings, _, pooled_text_embeddings, _ = pipe.encode_prompt([prompt])
    text_embeddings = text_embeddings.clone().detach()
    pooled_text_embeddings = pooled_text_embeddings.clone().detach()

    text_encoder = SDXLTextEncoder(pipe=pipe)
    embedding_calibrator = DynamicDOS(
        pipe=pipe, text_encoder=text_encoder, device=args.device,
        schedule_type=args.schedule_type, tau=args.tau
    )

    base_prompt_embeds, base_pooled_embeds, separation_prompt, separation_pooled = embedding_calibrator.compute_separation_vectors(
        target_prompt=prompt,
        target_objects=objects,
        text_embeddings=text_embeddings,
        pooled_text_embeddings=pooled_text_embeddings,
        separating_object_embedding=not args.disable_separating_object_embedding,
        separating_eot_embedding=not args.disable_separating_eot_embedding,
        separating_pooled_embedding=not args.disable_separating_pooled_embedding,
        lambda_sep=args.lambda_sep,
    )

    callback = embedding_calibrator.make_callback_on_step_end(
        base_prompt_embeds, base_pooled_embeds, separation_prompt, separation_pooled,
        num_inference_steps=50
    )

    generator = torch.Generator(args.device).manual_seed(seed)
    return pipe(
        prompt_embeds=base_prompt_embeds,
        pooled_prompt_embeds=base_pooled_embeds,
        generator=generator,
        callback_on_step_end=callback,
        callback_on_step_end_tensor_inputs=["prompt_embeds", "add_text_embeds"]
    ).images[0]


def main():
    args = parse_args()

    entries = read_prompt_entries(args.prompts_txt)
    if not entries:
        raise ValueError("No valid prompts found in the provided text file.")

    if args.num_prompts is not None:
        if args.num_prompts <= 0:
            raise ValueError("--num_prompts must be a positive integer.")
        entries = entries[: min(args.num_prompts, len(entries))]

    output_root = Path(args.output_path)
    sdxl_dir = output_root / args.base_subdir
    dydos_dir = output_root / args.dydos_subdir
    sdxl_dir.mkdir(parents=True, exist_ok=True)
    dydos_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded {len(entries)} prompts from {args.prompts_txt}")
    print(f"Saving SDXL images to: {sdxl_dir}")
    print(f"Saving DyDOS-SDXL images to: {dydos_dir}")

    pipe = load_sdxl_pipeline(args.device)

    skipped_count = 0
    saved_count = 0
    for idx, (prompt, objects, line_number) in enumerate(entries):
        if len(objects) < 2:
            print(
                f"[WARN] Line {line_number}: could not infer >=2 exact objects for DyDOS. "
                "Skipping this line and continuing. "
                "Use 'prompt || obj1, obj2' format for explicit objects."
            )
            skipped_count += 1
            continue

        filename = f"{prompt}_{saved_count:05d}.png"
        sdxl_path = sdxl_dir / filename
        dydos_path = dydos_dir / filename

        if (not args.overwrite) and sdxl_path.exists() and dydos_path.exists():
            print(f"[{idx + 1}/{len(entries)}] {filename} exists, skipping.")
            continue

        current_seed = args.seed + idx
        print(f"[{idx + 1}/{len(entries)}] Generating: {filename}")
        print(f"  Prompt: {prompt}")

        if not sdxl_path.exists() or args.overwrite:
            image_sdxl = generate_sdxl(pipe, prompt=prompt, seed=current_seed, device=args.device)
            image_sdxl.save(sdxl_path)
        else:
            image_sdxl = None

        print(f"  DyDOS objects: {objects}")
        image_dydos = generate_dos_sdxl(
            pipe,
            prompt=prompt,
            objects=objects,
            seed=current_seed,
            args=args,
        )
        image_dydos.save(dydos_path)
        saved_count += 1

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Done. Saved {saved_count} prompt pairs. Skipped {skipped_count} lines.")


if __name__ == "__main__":
    main()