import os
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output_root', type=str, default='results')
parser.add_argument('--save_name', type=str, required=True)
parser.add_argument('--object_filepaths', type=str, required=True, nargs='+')
parser.add_argument('--lora_path', type=str, default='checkpoints/lora_weights')
parser.add_argument('--rank', type=int, default=128)
parser.add_argument('--system_prompt_path', type=str, default='system_prompts/prompt_proposal.txt')
parser.add_argument('--user_character', type=str, default=None)
parser.add_argument('--translate_ratio', type=float, default=0.8)
parser.add_argument('--object_scale', type=float, default=0.8)
parser.add_argument('--shadow_length_min', type=float, default=0.6)
parser.add_argument('--shadow_length_max', type=float, default=1.3)
parser.add_argument('--elevation_count', type=int, default=4)
parser.add_argument('--azimuth_count', type=int, default=12)
parser.add_argument('--min_sketch_region_size', type=int, default=30)
parser.add_argument('--condition_type', type=str, default='canny', choices=['object_shadow', 'canny'])
parser.add_argument('--generation_type', type=str, default='full', choices=['full', 'partial'])
parser.add_argument('--stroke_image_dir', type=str, default='reverse', choices=['reverse', 'keyframes'])
parser.add_argument('--script_name', type=str, default='main')

# Object placement
parser.add_argument('--sample_distribution', action='store_true')
parser.add_argument('--optimize_object_params', action='store_true')
parser.add_argument('--num_iterations', type=int, default=200)
parser.add_argument('--render_count', type=int, default=60, help='Number of renders for each object to fit the distribution')
parser.add_argument('--use_gravity', action='store_true')

# process line drawings
parser.add_argument('--mask_width', type=int, default=3)
parser.add_argument('--process_kernel_size', type=int, default=3)

args = parser.parse_args()
command = f"python {args.script_name}.py \
        --lora_path {args.lora_path} \
        --rank {args.rank} \
        --translate_ratio {args.translate_ratio} \
        --internal_rotation \
        --object_scale {args.object_scale} \
        --output_root {args.output_root} \
        --shadow_length_min {args.shadow_length_min} \
        --shadow_length_max {args.shadow_length_max} \
        --elevation_count {args.elevation_count} \
        --azimuth_count {args.azimuth_count} \
        --object_filepath {' '.join(args.object_filepaths)} \
        --experiment_name {args.save_name} \
        --min_sketch_region_size {args.min_sketch_region_size} \
        --mask_width {args.mask_width} \
        --process_kernel_size {args.process_kernel_size} \
        --generation_type {args.generation_type} \
        --condition_type {args.condition_type} \
        --stroke_image_dir {args.stroke_image_dir} \
        --num_iterations {args.num_iterations} \
        --do_rank \
        --do_vqa_score"

if args.user_character:
    command += f" --user_character '{args.user_character}' --user_prompt_path {args.system_prompt_path}"
else:
    command += f" --system_prompt_path {args.system_prompt_path}"

if args.optimize_object_params:
    command += f" --optimize_object_params"

if args.sample_distribution:
    command += f" --sample_distribution --render_count {args.render_count}"

if args.use_gravity:
    command += f" --use_gravity"

subprocess.run(command, shell=True)
