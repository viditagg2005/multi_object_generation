import os
import glob
import argparse
import json
from hpsv3 import HPSv3RewardInferencer

def parse_args():
    parser = argparse.ArgumentParser(description="HPSv3 Batch Scoring Script")
    parser.add_argument("--target_dir", type=str, required=True, help="Directory containing generated images to score")
    parser.add_argument("--output_file", type=str, required=True, help="File to store the JSON results")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run HPSv3 inferencer on")
    parser.add_argument("--batch_size", type=int, default=1, help="Number of images to process simultaneously")
    return parser.parse_args()

def extract_prompt(filename):
    # Depending on format: "{prompt}_{index:05d}.png" or "{prompt}_{seed}.png"
    base = filename.replace(".png", "")
    parts = base.split("_")
    
    # If there's an index like 00000 or seed at the end, drop it
    if len(parts) > 1 and parts[-1].isdigit():
        return "_".join(parts[:-1])
    return base

def main():
    args = parse_args()
    
    print(f"Initializing HPSv3 Reward Inferencer on {args.device}...")
    inferencer = HPSv3RewardInferencer(device=args.device)
    
    image_files = glob.glob(os.path.join(args.target_dir, "*.png"))
    if not image_files:
        print(f"No PNG images found in {args.target_dir}")
        return
        
    print(f"Found {len(image_files)} images. Starting batch scoring...")
    
    results = []
    
    # Process the images in batches
    for i in range(0, len(image_files), args.batch_size):
        batch_files = image_files[i:i + args.batch_size]
        batch_prompts = [extract_prompt(os.path.basename(f)) for f in batch_files]
        
        try:
            # Pass the entire batch of prompts and image paths 
            rewards = inferencer.reward(prompts=batch_prompts, image_paths=batch_files)
            
            # Iterate through the returned scores to match them back to their filenames
            for j, img_path in enumerate(batch_files):
                filename = os.path.basename(img_path)
                # Extract the 'mu' and 'sigma' from the 2-element tensor [mu, sigma]
                mu = rewards[j][0].item()
                sigma = rewards[j][1].item()
                
                results.append({
                    "image": filename,
                    "prompt": batch_prompts[j],
                    "hpsv3_score": mu,
                    "sigma": sigma
                })
                print(f"Scored {filename}: mu={mu:.4f}, sigma={sigma:.4f}")
                
        except Exception as e:
            print(f"Error scoring batch starting at index {i}: {e}")
            
    if results:
        avg_score = sum(r['hpsv3_score'] for r in results) / len(results)
        print(f"\nAverage HPSv3 Score: {avg_score:.4f}")
        
        # Save results to json file
        output_data = {
            "average_hpsv3_score": avg_score,
            "total_images": len(results),
            "results": results
        }
        
        with open(args.output_file, 'w') as f:
            json.dump(output_data, f, indent=4)
        print(f"Saved detailed results to {args.output_file}")

if __name__ == "__main__":
    main()