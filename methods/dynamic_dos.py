import abc
import math
import torch
import configs.envs
import torch.nn.functional as F
from itertools import permutations
from utils.parser import get_word_index, get_object_indices
from configs.template import VISUAL_ATTRIBUTE, BACKGROUND



def get_inclination_profile(obj, text_encoder, prompts):
    text_batch, pooled_batch = text_encoder.encode(prompts)

    obj_profile = []
    eot_profile = []
    pooled_profile = []
    base_emb = text_batch[0]
    base_eot = None
    base_pooled = pooled_batch[0]
    for i, prompt in enumerate(prompts):
        idxs_obj = get_word_index(prompts[0].lower(), obj.lower(), text_encoder.tokenizer)
        idxs_attr_obj = get_word_index(prompts[i].lower(), obj.lower(), text_encoder.tokenizer)
        emb_obj = base_emb[idxs_obj]
        emb_attr_obj = text_batch[i, idxs_attr_obj]
        emb_obj = emb_obj.mean(dim=0)
        emb_attr_obj = emb_attr_obj.mean(dim=0)
        obj_profile.append(F.cosine_similarity(emb_obj, emb_attr_obj, dim=0))

        eot_idx = len(text_encoder.tokenizer.encode(prompt)) - 1
        eot_emb = text_batch[i, eot_idx]
        if i == 0:
            base_eot = eot_emb
            continue
        if base_eot is None:
            raise RuntimeError("base_eot was not initialized before cosine similarity")
        cos_eot = F.cosine_similarity(base_eot, eot_emb, dim=0)
        cos_pooled = F.cosine_similarity(base_pooled, pooled_batch[i], dim=0)
        eot_profile.append(cos_eot)
        pooled_profile.append(cos_pooled)

    return torch.stack(obj_profile).clone().detach(), torch.stack(eot_profile).clone().detach(), torch.stack(pooled_profile).clone().detach()


def get_visual_attribute_inclination(obj, text_encoder):
    print(f"Calculating visual_attribute_inclination for '{obj}'...")
    prompts = [f"a {obj}"] + [f"a {attr} {obj}" for attr in VISUAL_ATTRIBUTE]
    return get_inclination_profile(obj, text_encoder, prompts)


def get_background_inclination(obj, text_encoder):
    print(f"Calculating background_inclination for '{obj}'...")
    prompts = [f"a {obj}"] + [f"{scene}, there is a {obj}" for scene in BACKGROUND]
    return get_inclination_profile(obj, text_encoder, prompts)


def z_score_normalization(vector):
    mean = vector.mean()
    std = vector.std(unbiased=False)
    return (vector - mean) / std


def sigmoid_steep(x, T=1.0, center=0.5):
    return torch.sigmoid((x - center) / T)


class TextEncoder(abc.ABC):
    def __init__(self, pipe, tokenizer_attr='tokenizer'):
        self.pipe = pipe
        self.tokenizer = getattr(pipe, tokenizer_attr)

    @abc.abstractmethod
    def encode(self, prompt: str):
        """
        Encode a single prompt string.
        Returns: (text_embeddings, pooled_embeddings)
        """
        pass

    @abc.abstractmethod
    def pre_encode(self, text_embedings):
        pass

    @abc.abstractmethod
    def post_encode(self, text_embedings):
        pass


class SDXLTextEncoder(TextEncoder):
    def __init__(self, pipe, tokenizer_attr='tokenizer'):
        super().__init__(pipe, tokenizer_attr)
        self.sigmoid_temperature = 0.6

        self.sigmoid_center_visual_sim_obj = 0.5550
        self.sigmoid_center_visual_sim_eot = 0.5474
        self.sigmoid_center_visual_sim_pooled = 0.5366

        self.sigmoid_center_background_dist_obj = 0.1592
        self.sigmoid_center_background_dist_eot = 0.3862
        self.sigmoid_center_background_dist_pooled = 0.5835

    def encode(self, prompt: str):
        with torch.no_grad():
            out = self.pipe.encode_prompt(prompt)
        prompt_embedding = out[0].clone().detach()
        pooled_prompt_embedding = out[2].clone().detach()
        return prompt_embedding, pooled_prompt_embedding

    def pre_encode(self, text_embeddigs):
        return text_embeddigs

    def post_encode(self, text_embeddings):
        return text_embeddings


class SD3_5TextEncoder(TextEncoder):
    def __init__(self, pipe, tokenizer_attr='tokenizer'):
        super().__init__(pipe, tokenizer_attr)
        self.sigmoid_temperature = 0.6

        self.sigmoid_center_visual_sim_obj = 0.5536
        self.sigmoid_center_visual_sim_eot = 0.5473
        self.sigmoid_center_visual_sim_pooled = 0.6168

        self.sigmoid_center_background_dist_obj = 0.1705
        self.sigmoid_center_background_dist_eot = 0.3877
        self.sigmoid_center_background_dist_pooled = 0.4325

    def encode(self, prompt: str):
        prompt_embed, pooled_prompt_embed = self.pipe._get_clip_prompt_embeds(
            prompt=prompt,
            clip_model_index=0
        )
        prompt_2_embed, pooled_prompt_2_embed = self.pipe._get_clip_prompt_embeds(
            prompt=prompt,
            clip_model_index=1
        )
        prompt_embedding = torch.cat([prompt_embed, prompt_2_embed], dim=-1)
        pooled_prompt_embedding = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)
        return prompt_embedding, pooled_prompt_embedding

    def pre_encode(self, text_embeddings):
        self.text_embeddings_clip_and_t5 = text_embeddings.clone().detach()
        text_embeddings = self.text_embeddings_clip_and_t5[:, :77, :2048]  # only use CLIP portion
        return text_embeddings

    def post_encode(self, text_embeddings):
        self.text_embeddings_clip_and_t5[:, :77, :2048] = text_embeddings.clone().detach()
        text_embeddings = self.text_embeddings_clip_and_t5
        return text_embeddings


class DynamicDOS:
    SUPPORTED = {"constant", "linear", "cosine", "step"}

    def __init__(self, pipe, text_encoder, device, schedule_type="constant", tau=None):
        self.pipe = pipe
        self.text_encoder = text_encoder
        self.device = device
        if schedule_type not in self.SUPPORTED:
            raise ValueError(f"Invalid schedule_type: {schedule_type}. Expected one of: {sorted(self.SUPPORTED)}")
        self.schedule_type = schedule_type
        self.tau = tau

    def _resolve_tau(self, total_steps):
        if self.tau is None:
            return 0.5
        if self.tau > 1.0:
            denom = max(total_steps - 1, 1)
            return max(0.0, min(1.0, self.tau / denom))
        return max(0.0, min(1.0, self.tau))

    def alpha_from_step(self, step_index, total_steps):
        import math
        denom = max(total_steps - 1, 1)
        normalized_t = max(0.0, min(1.0, 1.0 - (float(step_index) / float(denom))))

        if self.schedule_type == "constant":
            return 1.0
        if self.schedule_type == "linear":
            return normalized_t
        if self.schedule_type == "cosine":
            return 0.5 * (1.0 - math.cos(math.pi * normalized_t))

        tau = self._resolve_tau(total_steps)
        return 1.0 if normalized_t > tau else 0.0

    def validate_prompt(self, target_prompt, target_objects):
        if type(target_prompt) == list:
            target_prompt = target_prompt[0]

        print("====================== start DOS =========================")
        print(f"target prompt: '{target_prompt}'")
        for idx, target_prompt_object in enumerate(target_objects):
            print(f"target_prompt_obj[{idx}]: '{target_prompt_object}'")

        if len(target_objects) < 2:
            raise ValueError("The number of object indices is < 2")

        return target_prompt

    def collect_object_embeddings(self, target_objects, text_embeddings, object_indices):
        objs = []
        objs_pure = []
        objs_pure_eot = []
        objs_pure_pooled = []
        for obj_idx, obj in enumerate(target_objects):
            prompt_obj_pure = [f"a {obj}"]
            text_embeddings_obj_pure, pooled_embedding_obj_pure = self.text_encoder.encode(prompt_obj_pure)

            obj_token_indices = object_indices[obj_idx]
            obj_pure_token_indices = get_word_index(prompt_obj_pure[0], obj, self.text_encoder.tokenizer)
            eot_token_idx = len(self.text_encoder.tokenizer.encode(prompt_obj_pure[0])) - 1
            if len(obj_token_indices) == 0:
                raise ValueError("Word indices error")

            obj_embeddings = text_embeddings[0, obj_token_indices].clone().detach()
            obj_pure_embeddings = text_embeddings_obj_pure[0, obj_pure_token_indices].clone().detach()
            obj_pure_eot_embedding = text_embeddings_obj_pure[0, eot_token_idx].clone().detach()
            obj_pure_pooled_embedding = pooled_embedding_obj_pure[0].clone().detach()

            averaged_obj_embedding = torch.mean(obj_embeddings, dim=0)
            averaged_obj_pure_embedding = torch.mean(obj_pure_embeddings, dim=0)

            objs.append(averaged_obj_embedding)
            objs_pure.append(averaged_obj_pure_embedding)
            objs_pure_eot.append(obj_pure_eot_embedding)
            objs_pure_pooled.append(obj_pure_pooled_embedding)
            torch.cuda.empty_cache()

        objs = torch.stack(objs)
        objs_pure = torch.stack(objs_pure)
        objs_pure_eot = torch.stack(objs_pure_eot)
        objs_pure_pooled = torch.stack(objs_pure_pooled)
        return objs, objs_pure, objs_pure_eot, objs_pure_pooled

    def collect_background_inclinations(self, target_objects):
        background_inclinations = {}
        for obj in target_objects:
            obj_profile, eot_profile, pooled_profile = get_background_inclination(obj, self.text_encoder)
            obj_profile = z_score_normalization(obj_profile)
            eot_profile = z_score_normalization(eot_profile)
            pooled_profile = z_score_normalization(pooled_profile)
            background_inclinations[obj] = (obj_profile, eot_profile, pooled_profile)
        return background_inclinations

    def collect_visual_attribute_inclinations(self, target_objects):
        visual_attribute_inclinations = {}
        for obj in target_objects:
            obj_profile, eot_profile, pooled_profile = get_visual_attribute_inclination(obj, self.text_encoder)
            obj_profile = z_score_normalization(obj_profile)
            eot_profile = z_score_normalization(eot_profile)
            pooled_profile = z_score_normalization(pooled_profile)
            visual_attribute_inclinations[obj] = (obj_profile, eot_profile, pooled_profile)
        return visual_attribute_inclinations

    def update_text_embeddings(self, target_prompt, target_objects, text_embeddings, pooled_text_embeddings,
                               **config):
        text_embeddings = self.text_encoder.pre_encode(text_embeddings)
        target_prompt = self.validate_prompt(target_prompt, target_objects)

        object_indices = get_object_indices(target_prompt, target_objects, self.text_encoder.tokenizer)

        _, objs_pure, _, _ = self.collect_object_embeddings(target_objects, text_embeddings, object_indices)

        visual_attribute_inclinations = self.collect_visual_attribute_inclinations(target_objects)
        background_inclinations = self.collect_background_inclinations(target_objects)

        k = len(target_objects)
        for object_pair in permutations(target_objects, 2):
            obj0_idx = target_objects.index(object_pair[0])
            obj1_idx = target_objects.index(object_pair[1])
            print("------------------------------------------------------------------")
            print(f"Object pair for injecting DOS vectors: '{object_pair[0]}' and '{object_pair[1]}'")
            print(f"Index for '{object_pair[0]}': {obj0_idx}")
            print(f"Index for '{object_pair[1]}': {obj1_idx}")

            visual_attribute_inclination_obj0 = visual_attribute_inclinations[object_pair[0]]
            visual_attribute_inclination_obj1 = visual_attribute_inclinations[object_pair[1]]
            visual_sim_obj = F.cosine_similarity(visual_attribute_inclination_obj0[0], visual_attribute_inclination_obj1[0], dim=0)
            visual_sim_eot = F.cosine_similarity(visual_attribute_inclination_obj0[1], visual_attribute_inclination_obj1[1], dim=0)
            visual_sim_pooled = F.cosine_similarity(visual_attribute_inclination_obj0[2], visual_attribute_inclination_obj1[2], dim=0)

            background_inclination_obj0 = background_inclinations[object_pair[0]]
            background_inclination_obj1 = background_inclinations[object_pair[1]]
            background_dist_obj = 1 - F.cosine_similarity(background_inclination_obj0[0], background_inclination_obj1[0], dim=0)
            background_dist_eot = 1 - F.cosine_similarity(background_inclination_obj0[1], background_inclination_obj1[1], dim=0)
            background_dist_pooled = 1 - F.cosine_similarity(background_inclination_obj0[2], background_inclination_obj1[2], dim=0)

            # create separation vector for object
            visual_sim_obj = sigmoid_steep(visual_sim_obj, T=self.text_encoder.sigmoid_temperature, center=self.text_encoder.sigmoid_center_visual_sim_obj)
            background_dist_obj = sigmoid_steep(background_dist_obj, T=self.text_encoder.sigmoid_temperature, center=self.text_encoder.sigmoid_center_background_dist_obj)
            alpha_obj = config.get('lambda_sep', 1.0) * torch.max(torch.stack([visual_sim_obj, background_dist_obj])) / (k - 1)
            separating_vector_obj = alpha_obj * (objs_pure[obj0_idx] - objs_pure[obj1_idx])

            prompt_sep = [f"a {object_pair[0]} separated from a {object_pair[1]}"]
            prompt_mix = [f"a {object_pair[0]} mixed with a {object_pair[1]}"]

            prompt_embedding_sep, pooled_prompt_embedding_sep = self.text_encoder.encode(prompt_sep)
            prompt_embedding_mix, pooled_prompt_embedding_mix = self.text_encoder.encode(prompt_mix)

            prompt_sep_eot_token_idx = len(self.text_encoder.tokenizer.encode(prompt_sep[0])) - 1
            prompt_mix_eot_token_idx = len(self.text_encoder.tokenizer.encode(prompt_mix[0])) - 1

            # create separation vectors for eot/pooled
            visual_sim_eot = sigmoid_steep(visual_sim_eot, T=self.text_encoder.sigmoid_temperature, center=self.text_encoder.sigmoid_center_visual_sim_eot)
            background_dist_eot = sigmoid_steep(background_dist_eot, T=self.text_encoder.sigmoid_temperature, center=self.text_encoder.sigmoid_center_background_dist_eot)
            visual_sim_pooled = sigmoid_steep(visual_sim_pooled, T=self.text_encoder.sigmoid_temperature, center=self.text_encoder.sigmoid_center_visual_sim_pooled)
            background_dist_pooled = sigmoid_steep(background_dist_pooled, T=self.text_encoder.sigmoid_temperature, center=self.text_encoder.sigmoid_center_background_dist_pooled)
            alpha_eot = config.get('lambda_sep', 1.0) * torch.max(torch.stack([visual_sim_eot, background_dist_eot])) / (k - 1)
            alpha_pooled = config.get('lambda_sep', 1.0) * torch.max(torch.stack([visual_sim_pooled, background_dist_pooled])) / (k - 1)
            separating_vector_eot = alpha_eot * (prompt_embedding_sep[0, prompt_sep_eot_token_idx] - prompt_embedding_mix[0, prompt_mix_eot_token_idx])
            separating_vector_pooled = alpha_pooled * (pooled_prompt_embedding_sep - pooled_prompt_embedding_mix)

            eot_token_idx = len(self.text_encoder.tokenizer.encode(target_prompt)) - 1

            if config.get('separating_object_embedding', True):
                obj_token_indices = object_indices[obj0_idx]
                print(f"Injecting separation vector on object (token indices: {obj_token_indices}): {alpha_obj:.4f} * (['a {object_pair[0]}'] - ['a {object_pair[1]}'])")
                text_embeddings[0, obj_token_indices] += separating_vector_obj.clone().detach()

            if config.get('separating_eot_embedding', True):
                print(f"Injecting separation vector on EOT (token index: {eot_token_idx}): {alpha_eot:.4f} * ({prompt_sep} - {prompt_mix})")
                text_embeddings[0, eot_token_idx] += separating_vector_eot.clone().detach()

            if config.get('separating_pooled_embedding', True):
                print(f"Injecting separation vector on pooled: {alpha_pooled:.4f} * ({prompt_sep} - {prompt_mix})")
                pooled_text_embeddings += separating_vector_pooled.clone().detach()

            del separating_vector_eot, separating_vector_pooled
            del pooled_prompt_embedding_sep, pooled_prompt_embedding_mix
            del prompt_embedding_sep, prompt_embedding_mix
            torch.cuda.empty_cache()

        text_embeddings = self.text_encoder.post_encode(text_embeddings)
        return text_embeddings.clone().detach(), pooled_text_embeddings.clone().detach()



    def compute_separation_vectors(self, target_prompt, target_objects, text_embeddings, pooled_text_embeddings, **config):
        base_prompt_embeds = text_embeddings.clone().detach()
        base_pooled_embeds = pooled_text_embeddings.clone().detach()

        static_prompt_embeds, static_pooled_embeds = self.update_text_embeddings(
            target_prompt=target_prompt,
            target_objects=target_objects,
            text_embeddings=text_embeddings.clone().detach(),
            pooled_text_embeddings=pooled_text_embeddings.clone().detach(),
            **config,
        )

        separation_prompt = static_prompt_embeds - base_prompt_embeds
        separation_pooled = static_pooled_embeds - base_pooled_embeds
        return base_prompt_embeds, base_pooled_embeds, separation_prompt, separation_pooled

    def _inject_conditional_embeddings(self, current_embeddings, base_embeddings, separation_vector, alpha):
        scaled = base_embeddings + (alpha * separation_vector)

        if current_embeddings.shape[0] == scaled.shape[0] * 2:
            updated = current_embeddings.clone()
            updated[scaled.shape[0]:] = scaled.to(device=current_embeddings.device, dtype=current_embeddings.dtype)
            return updated

        return scaled.to(device=current_embeddings.device, dtype=current_embeddings.dtype)

    def make_callback_on_step_end(self, base_prompt_embeds, base_pooled_embeds, separation_prompt, separation_pooled, num_inference_steps=50):
        def _callback_on_step_end(pipe, step_index, timestep, callback_kwargs):
            next_step = min(step_index + 1, num_inference_steps - 1)
            alpha = self.alpha_from_step(next_step, num_inference_steps)

            if "prompt_embeds" in callback_kwargs:
                callback_kwargs["prompt_embeds"] = self._inject_conditional_embeddings(
                    current_embeddings=callback_kwargs["prompt_embeds"],
                    base_embeddings=base_prompt_embeds,
                    separation_vector=separation_prompt,
                    alpha=alpha,
                )

            if "pooled_prompt_embeds" in callback_kwargs:
                callback_kwargs["pooled_prompt_embeds"] = self._inject_conditional_embeddings(
                    current_embeddings=callback_kwargs["pooled_prompt_embeds"],
                    base_embeddings=base_pooled_embeds,
                    separation_vector=separation_pooled,
                    alpha=alpha,
                )
                
            if "add_text_embeds" in callback_kwargs:
                callback_kwargs["add_text_embeds"] = self._inject_conditional_embeddings(
                    current_embeddings=callback_kwargs["add_text_embeds"],
                    base_embeddings=base_pooled_embeds,
                    separation_vector=separation_pooled,
                    alpha=alpha,
                )
            return callback_kwargs

        return _callback_on_step_end
