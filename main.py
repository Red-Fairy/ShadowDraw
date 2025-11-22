from PIL import Image, ImageOps
from datetime import datetime
import os
import base64
import cv2
import numpy as np
import io
import socket
import json
import argparse
import math
from typing import List
from scipy.ndimage import binary_fill_holes, generate_binary_structure, label
from src import make_grid, encode_image, fill_small_holes
from tqdm import tqdm
from openai import OpenAI
from src import get_imagereward_scores, get_clip_scores, get_hps_scores, compute_irregularity, DistributionFitter
from scipy.special import erf
import subprocess
import time
from torch.utils.tensorboard import SummaryWriter

'''
Remember to set the OPENAI_API_KEY in the environment variables.
'''

def animate_object(global_root, object_filepaths, blender_path, initial_height=0.1, simulation_frames=250, try_put_down=True, object_scale=0.8, combine_multiple_objects=False):

    for i, object_filepath in enumerate(object_filepaths):
        if object_filepath.endswith('glb'):
            if not os.path.exists(object_filepath.replace('.glb', '.obj')):
                subprocess.run([blender_path, "--background", "--python", "src/convert_obj.py", "--", "--object_path", object_filepath])
            object_filepaths[i] = object_filepath.replace('.glb', '.obj')

    if len(object_filepaths) == 1 or not combine_multiple_objects:
        for i, object_filepath in enumerate(object_filepaths):
            animate_save_path = os.path.join(global_root, f'animate_{i}.json')
            if os.path.exists(animate_save_path):
                print(f"Animate file already exists for {object_filepath}")
                return
            params = {
                "object_filepath": object_filepath,
                "initial_height": initial_height,
                "simulation_frames": simulation_frames,
                "do_animate": True,
                "animate_save_path": animate_save_path,
                "try_put_down": try_put_down,
            }
            cmd = [blender_path, 
                '--background', 
                '--python', 'server/blender.py', 
                '--', 
                '--json_data', 
                json.dumps(params)]
            subprocess.run(cmd)
    else:
        combined_object_filepath = os.path.join(global_root, 'combined_objects.obj')
        if os.path.exists(combined_object_filepath):
            print(f"Combined object file already exists")
            return
        params = {
            "object_filepath": object_filepaths,
            "object_scale": object_scale,
            "output_root": global_root,
            "try_put_down": try_put_down,
            "do_animate": True,
        }
        cmd = [blender_path, 
            '--background', 
            '--python', 'server/blender.py', 
            '--', 
            '--json_data', 
            json.dumps(params)]
        subprocess.run(cmd)
    return

def compute_light_elevations(output_root, object_filepaths, elevation_count, object_scale, translate_ratio, shadow_length_min, shadow_length_max, use_gravity, blender_path, try_put_down=True):

    if os.path.exists(os.path.join(output_root, 'light_elevations.json')):
        print('Light elevations already computed')
    else:
        params = {
            "object_filepath": object_filepaths,
            "object_scale": object_scale,
            "elevation": elevation_count,
            "output_root": output_root,
            "translate_ratio": translate_ratio,
            "shadow_length_min": shadow_length_min,
            "shadow_length_max": shadow_length_max,
            "compute_light_elevation": True,
            "try_put_down": try_put_down,
            "use_gravity": use_gravity,
        }

        cmd = [blender_path, 
            '--background', 
            '--python', 'server/blender.py', 
            '--', 
            '--json_data', 
            json.dumps(params)]
        subprocess.run(cmd)
    return

def compute_irregularity_for_sampling(output_root, object_filepaths, render_count, object_scale, translate_ratio, use_gravity):

    for i, object_filepath in enumerate(object_filepaths):
        irregularity_save_path = os.path.join(output_root, f'irregularity_{i}.json')
        if os.path.exists(irregularity_save_path):
            print(f'Irregularity already computed for {object_filepath}')
            continue
        light_elevations = json.load(open(os.path.join(output_root, 'light_elevations.json')))
        cmd = ['python', 'src/render_shadow.py', 
                '--object_filepath', object_filepath,
                '--output_root', output_root,
                '--object_index', str(i),
                '--scale_target', str(object_scale),
                '--translate_ratio', str(translate_ratio),
                '--azimuths_per_elevation', str(render_count)]
        if use_gravity:
            cmd.append('--use_gravity')
        else:
            cmd.append('--disable_put_down')
        cmd.append('--light_elevations')
        cmd.extend([str(light_elevation) for light_elevation in light_elevations for _ in range(render_count)])

        subprocess.run(cmd)

        object_shadow_binary_root = os.path.join(output_root, f'object_shadow_binary_{i}')
        irregularity_dict = compute_irregularity(object_shadow_binary_root)
        with open(os.path.join(output_root, f'irregularity_{i}.json'), 'w') as f:
            json.dump(irregularity_dict, f)

    return

def optimize_object_params(output_root, object_filepaths, azimuth_count, object_scale, translate_ratio, num_iterations, use_gravity):

    if os.path.exists(os.path.join(output_root, 'object_params_optimized.json')):
        print('Object parameters already optimized')
    else:
        light_elevations = json.load(open(os.path.join(output_root, 'light_elevations.json')))
        cmd = ['python', 'src/optimize_object_params.py', 
                '--object_filepath', object_filepaths[0],
                '--output_root', output_root,
                '--num_iterations', str(num_iterations),
                '--translate_ratio', str(translate_ratio),
                '--scale_target', str(object_scale),
                '--azimuths_per_elevation', str(azimuth_count)]

        cmd.append('--light_elevations')
        cmd.extend([str(light_elevation) for light_elevation in light_elevations for _ in range(azimuth_count)])

        if use_gravity:
            cmd.append('--use_gravity')
        else:
            cmd.append('--disable_put_down')

        subprocess.run(cmd)

    return

def render_image(output_root, object_filepaths, azimuth_count, elevation_count, object_scale, translate_ratio, render_resolution, blender_path, 
                 shadow_length_min, shadow_length_max, read_scene_params=False,
                 synthetic_rotation=False, multi_object_config=None, internal_rotation=False, sample_distribution=False, use_gravity=False, try_put_down=True):

    params = {
        "object_filepath": object_filepaths,
        "object_scale": object_scale,
        "azimuth": azimuth_count if not synthetic_rotation else 1,
        "elevation": elevation_count,
        "output_root": output_root,
        'resolution': render_resolution,
        'apply_texture': False,
        'translate_ratio': translate_ratio,
        'dynamic_elevation': True,
        'shadow_length_min': shadow_length_min,
        'shadow_length_max': shadow_length_max,
        'multi_object_config': multi_object_config,
        'internal_rotation': internal_rotation,
        'sample_distribution': sample_distribution,
        'shadow_sampling': False,
        'read_scene_params': read_scene_params,
        'use_gravity': use_gravity,
        'try_put_down': try_put_down,
        'add_material': all('chill' in object_filepath.lower() for object_filepath in object_filepaths)
    }

    cmd = [blender_path, 
               '--background', 
               '--python', 'server/blender.py', 
               '--', 
               '--json_data', 
               json.dumps(params)]
    subprocess.run(cmd)

    if synthetic_rotation:
        print('Synthesizing rotation images')
        rotation_degrees = np.linspace(0, 360, azimuth_count, endpoint=False).astype(np.int32).tolist()
        object_shadow_image_root = os.path.join(output_root, 'object_shadow')
        object_image_root = os.path.join(output_root, 'object')
        image_names = os.listdir(object_shadow_image_root)
        for image_name in image_names:
            object_shadow_image = Image.open(os.path.join(object_shadow_image_root, image_name))
            object_image = Image.open(os.path.join(object_image_root, image_name))
            for rotation_degree in rotation_degrees:
                rotated_object_shadow_image = object_shadow_image.rotate(rotation_degree, fillcolor=(255, 255, 255, 255))
                rotated_object_image = object_image.rotate(rotation_degree, fillcolor=0)
                rotated_object_shadow_image.save(os.path.join(object_shadow_image_root, image_name.replace('azi000', f'azi{rotation_degree:03d}')))
                rotated_object_image.save(os.path.join(object_image_root, image_name.replace('azi000', f'azi{rotation_degree:03d}')))

    return

def process_rendering(output_root, generation_resolution, dilate_size, max_shadow_hole_size, min_shadow_region_size, multi_object_config):

    object_shadow_image_root = os.path.join(output_root, 'object_shadow')
    object_image_root = os.path.join(output_root, 'object')
    reverse_image_root = os.path.join(output_root, 'reverse')
    canny_edge_image_root = os.path.join(output_root, 'canny_edge')
    shadow_image_root = os.path.join(output_root, 'shadow')

    os.makedirs(reverse_image_root, exist_ok=True)
    os.makedirs(canny_edge_image_root, exist_ok=True)
    os.makedirs(shadow_image_root, exist_ok=True)

    if len(os.listdir(reverse_image_root)) == len(os.listdir(object_shadow_image_root)):
        print('Processing already done')
        return

    image_names = os.listdir(object_shadow_image_root)

    for image_name in image_names:
        object_shadow_image_path = os.path.join(object_shadow_image_root, image_name)
        object_image_path = os.path.join(object_image_root, image_name)
        reverse_image_path = os.path.join(reverse_image_root, image_name)
        canny_edge_image_path = os.path.join(canny_edge_image_root, image_name)
        shadow_image_path = os.path.join(shadow_image_root, image_name)

        object_shadow_image = Image.open(object_shadow_image_path)
        object_image = Image.open(object_image_path)

        # get the object mask
        mask = object_image.split()[-1]
        mask = np.array(mask) == 0

        object_shadow_image.putalpha(Image.fromarray(mask))

        white_bg = Image.new('RGB', object_image.size, (255, 255, 255))
        white_bg.paste(object_shadow_image, (0, 0), object_shadow_image)

        shadow_image = np.array(white_bg)
        avg_values = np.mean(shadow_image, axis=2)

        threshold = np.mean(avg_values[mask])
        dark_mask = avg_values < threshold - 20

        if multi_object_config != "in_out":
            binary_shadow_image = np.where(dark_mask, 1, 0)
        else: # multi_object_config == "in_out"
            small_mask = Image.open(os.path.join(output_root, 'object_small', image_name)).split()[-1]
            small_mask = np.array(small_mask) == 255
            binary_shadow_image = np.where(np.logical_or(dark_mask, small_mask), 1, 0)

        # fill holes
        binary_shadow_image = fill_small_holes(binary_shadow_image, max_shadow_hole_size=max_shadow_hole_size)
        binary_shadow_image = binary_shadow_image.astype(np.uint8) * 255

        # Smooth the shadow mask
        mask_med = cv2.medianBlur(binary_shadow_image, ksize=5)
        mask_blur = cv2.GaussianBlur(mask_med, ksize=(7,7), sigmaX=0)

        # Erode and then dilate
        mask_eroded = cv2.erode(mask_blur, np.ones((5,5), np.uint8), iterations=1)
        mask_dilated = cv2.dilate(mask_eroded, np.ones((5,5), np.uint8), iterations=1)

        # back to binary
        _, binary_shadow_image = cv2.threshold(mask_dilated, 127, 255, cv2.THRESH_BINARY)

        # the mask should only contain one connected component, remove the small connected components
        labeled_regions, num_labels = label(binary_shadow_image, structure=generate_binary_structure(2, 1))
        region_sizes = []
        for i in range(1, num_labels + 1):
            region_mask = labeled_regions == i
            region_sizes.append(np.sum(region_mask))
        
        max_region_size = np.max(region_sizes)
        for i in range(1, num_labels + 1):
            if region_sizes[i - 1] < min_shadow_region_size and region_sizes[i - 1] != max_region_size:
                binary_shadow_image = np.where(labeled_regions == i, 0, binary_shadow_image)
        
        binary_shadow_image = 255 - binary_shadow_image
        binary_shadow_image = cv2.resize(binary_shadow_image, (generation_resolution, generation_resolution), interpolation=cv2.INTER_NEAREST)

        # save the shadow image
        Image.fromarray(binary_shadow_image).save(shadow_image_path)

        # extract the edge of the shadow image
        canny_image = cv2.Canny(binary_shadow_image, 50, 150)
        Image.fromarray(canny_image).save(canny_edge_image_path)

        # save the reverse image
        canny_image_np = cv2.dilate(canny_image.astype(np.uint8), np.ones((dilate_size, dilate_size), np.uint8), iterations=1)
        reverse_image = 255 - canny_image_np
        Image.fromarray(reverse_image).save(reverse_image_path)

    return

def create_prompt_gpt(output_root, system_prompt, prompt_resolution, stroke_image_dir='reverse', override_prompt=None):

    stroke_image_root = os.path.join(output_root, stroke_image_dir)
    prompt_root = os.path.join(output_root, 'prompt')   
    os.makedirs(prompt_root, exist_ok=True)
    image_names = os.listdir(stroke_image_root)

    if override_prompt:
        return [override_prompt] * len(image_names)

    # system_prompt = open(system_prompt_path).read()
    stroke_images = [Image.open(os.path.join(stroke_image_root, image_name)) for image_name in image_names]
    stroke_images_base64 = [encode_image(stroke_image, size=(prompt_resolution, prompt_resolution)) for stroke_image in stroke_images]

    prompt_texts = []
    client = OpenAI()

    pbar = tqdm(total=len(stroke_images_base64), desc='Generating prompts')
    for image_name, stroke_image_base64 in zip(image_names, stroke_images_base64):

        save_path = os.path.join(prompt_root, image_name.replace('.png', '.txt'))
        if os.path.exists(save_path):
            print(f'Prompt already exists for {image_name}')
            # read the prompt from the file
            with open(save_path, 'r') as f:
                prompt_text = f.read()
            prompt_texts.append(prompt_text)
            pbar.update(1)
            continue

        while True:
            try:
                response = client.responses.create(
                    model="gpt-4.1",
                    input=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": f"data:image/png;base64,{stroke_image_base64}",
                                    "detail": "high"
                                }
                            ],
                        }
                    ],
                    # temperature=0.5
                )
                prompt_text = response.output_text
                prompt_texts.append(prompt_text)
                break

            except Exception as e:
                print(f'Error generating prompt for {image_name}: {e}')
                time.sleep(5)
                continue

        with open(save_path, 'w') as f:
            f.write(prompt_text)

        pbar.update(1)

    return

def create_line_drawing(output_root, guidance_scale, num_inference_steps, generation_resolution, erode_pixels,
                        lora_path=None, pretrained_model_name_or_path=None, condition_type='canny', batch_size=8, rank=64):

    control_image_root = os.path.join(output_root, 'canny_edge') if condition_type == 'canny' else os.path.join(output_root, 'object_shadow')
    mask_image_root = os.path.join(output_root, 'object')

    image_names = os.listdir(control_image_root) 
    output_paths = [os.path.join(output_root, 'line_drawing', image_name) for image_name in image_names]
    prompt_filepaths = [os.path.join(output_root, 'prompt', image_name.replace('.png', '.txt')) for image_name in image_names]

    if all(os.path.exists(output_path) for output_path in output_paths):
        print('Line drawing already exists')
        return

    params = {
        "prompt": prompt_filepaths,
        "guidance_scale": guidance_scale,
        "control_image_path": control_image_root,
        "mask_image_path": mask_image_root,
        "num_inference_steps": num_inference_steps,
        'generation_resolution': generation_resolution,
        'output_paths': output_paths,
        'erode_pixels': erode_pixels,
        'batch_size': batch_size
    }
    assert lora_path is not None and pretrained_model_name_or_path is not None, "lora_path and pretrained_model_name_or_path must be provided"
    cmd = ['python', 
            'server/diffusion.py', 
            '--json_data', 
            json.dumps(params), 
            '--lora_path', 
            lora_path, 
            '--pretrained_model_name_or_path', 
            pretrained_model_name_or_path,
            '--rank',
            str(rank)
            ]
    subprocess.run(cmd)
        
    return
        
def process_line_drawing(output_root, mask_width, process_kernel_size, min_sketch_region_size=-1):
    
    line_drawing_root = os.path.join(output_root, 'line_drawing')
    processed_line_drawing_root = f'{line_drawing_root}_processed'
    shadow_image_root = os.path.join(output_root, 'shadow')

    image_names = os.listdir(line_drawing_root)
    line_drawing_paths = [os.path.join(line_drawing_root, image_name) for image_name in image_names]
    shadow_image_paths = [os.path.join(shadow_image_root, image_name) for image_name in image_names]
    
    os.makedirs(processed_line_drawing_root, exist_ok=True)
    processed_line_drawing_paths = [os.path.join(processed_line_drawing_root, image_name) for image_name in image_names]
    
    for line_drawing_path, shadow_image_path, processed_line_drawing_path in zip(line_drawing_paths, shadow_image_paths, processed_line_drawing_paths):

        generated_image = np.array(Image.open(line_drawing_path))
        shadow_image = np.array(Image.open(shadow_image_path).convert('L'))

        # get the boundary of the dark mask
        kernel_size = mask_width * 2 + 1
        dark_mask_dilated = cv2.dilate(shadow_image, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
        dark_mask_eroded = cv2.erode(shadow_image, np.ones((kernel_size, kernel_size), np.uint8), iterations=1)
        dark_mask = dark_mask_dilated - dark_mask_eroded

        # resize to the same size as the generation image
        dark_mask = cv2.resize(dark_mask, (generated_image.shape[1], generated_image.shape[0]), interpolation=cv2.INTER_NEAREST)
        dark_mask = np.where(dark_mask > 0, True, False)

        # apply the dark mask to the generation image, make the dark area white
        generated_image_masked = np.where(dark_mask[..., None], 255, generated_image)

        # dilate and erode the processed image to remove the noise
        generated_image_masked = cv2.dilate(generated_image_masked, np.ones((process_kernel_size, process_kernel_size), np.uint8), iterations=1)
        generated_image_masked = cv2.erode(generated_image_masked, np.ones((process_kernel_size, process_kernel_size), np.uint8), iterations=1)
        
        # reverse, then remove connected components with < min_sketch_region_size
        if min_sketch_region_size > 0:
            generated_image_binary = np.where(generated_image_masked.mean(axis=2) > 128, 0, 255) # binary image
            labeled_regions, num_labels = label(generated_image_binary, structure=generate_binary_structure(2, 1)) 

            # set regions with < min_sketch_region_size to 0
            for i in range(1, num_labels + 1):
                region_mask = labeled_regions == i
                if np.sum(region_mask) < min_sketch_region_size:
                    generated_image_masked = np.where(region_mask[..., None], 255, generated_image_masked)

        generated_image_processed = np.where(generated_image_masked.mean(axis=2)[..., None] < 128, generated_image, 255)

        # save the generation image
        Image.fromarray(generated_image_processed).save(processed_line_drawing_path)

def process_line_drawing_object_shadow(output_root, mask_width):
    
    line_drawing_root = os.path.join(output_root, 'line_drawing')
    processed_line_drawing_root = f'{line_drawing_root}_processed'
    shadow_stroke_root = os.path.join(output_root, 'reverse')
    if not os.path.exists(processed_line_drawing_root):
        os.system(f'cp -r {line_drawing_root} {processed_line_drawing_root}')

    image_names = os.listdir(line_drawing_root)

    line_drawing_paths = [os.path.join(processed_line_drawing_root, image_name) for image_name in image_names]
    shadow_stroke_paths = [os.path.join(shadow_stroke_root, image_name) for image_name in image_names]
    save_line_drawing_paths = [os.path.join(line_drawing_root, image_name) for image_name in image_names]
    
    for line_drawing_path, shadow_stroke_path, save_line_drawing_path in zip(line_drawing_paths, shadow_stroke_paths, save_line_drawing_paths):

        generated_image = np.array(Image.open(line_drawing_path))
        shadow_stroke_image = np.array(Image.open(shadow_stroke_path).convert('L'))
        shadow_stroke_image = cv2.erode(shadow_stroke_image, np.ones((mask_width, mask_width), np.uint8), iterations=1)
        processed_line_drawing = np.where(shadow_stroke_image[..., None] > 128, generated_image, 0)

        # save the generation image
        Image.fromarray(processed_line_drawing).save(save_line_drawing_path)

    return

def render_shadow_art(output_root, object_filepaths, elevation_count, azimuth_count, object_scale, translate_ratio, render_resolution, blender_path, 
                       shadow_length_min, shadow_length_max, multi_object_config, internal_rotation, use_gravity, try_put_down=True):

    params = {  
        "object_filepath": object_filepaths,
        "object_scale": object_scale,
        "translate_ratio": translate_ratio,
        "azimuth": azimuth_count,
        "elevation": elevation_count,
        "output_root": output_root,
        'resolution': render_resolution,
        'apply_texture': True,
        'dynamic_elevation': True,
        'shadow_length_min': shadow_length_min,
        'shadow_length_max': shadow_length_max,
        'multi_object_config': multi_object_config,
        'internal_rotation': internal_rotation,
        'use_gravity': use_gravity,
        'try_put_down': try_put_down,
        'add_material': all('chill' in object_filepath.lower() for object_filepath in object_filepaths)
    }

    cmd = [blender_path, 
            '--background', 
            '--python', 
            'server/blender.py', 
            '--', 
            '--json_data', 
            json.dumps(params)]
    subprocess.run(cmd)

    print(f'Success! Finished creating shadow art images')

def get_scores(output_root: str, metrics_list: list):
    line_drawing_root = os.path.join(output_root, 'line_drawing')
    line_drawing_processed_root = os.path.join(output_root, 'line_drawing_processed')
    shadow_art_root = os.path.join(output_root, 'shadow_art')
    image_names = os.listdir(line_drawing_root)
    line_drawing_paths = [os.path.join(line_drawing_root, image_name) for image_name in image_names]
    line_drawing_processed_paths = [os.path.join(line_drawing_processed_root, image_name) for image_name in image_names]
    # shadow_art_paths = [os.path.join(shadow_art_root, image_name) for image_name in image_names]
    prompts = [open(os.path.join(output_root, 'prompt', image_name.replace('.png', '.txt'))).readlines()[-1].strip() for image_name in image_names]

    for metric_dict in metrics_list:
        save_path = os.path.join(output_root, f'{metric_dict["metric_save_name"]}_scores.json')
        if os.path.exists(save_path):
            print(f'{metric_dict["metric_save_name"]} scores already exist')
            continue
        print(f'Computing {metric_dict["metric_save_name"]} scores')
        scores = metric_dict['fn'](line_drawing_paths, prompts, model_name=metric_dict['model_name'])
        scores_processed = metric_dict['fn'](line_drawing_processed_paths, prompts, model_name=metric_dict['model_name'])
        # prompts_shadow_art = [prompt.replace('A minimalist line drawing of', 'A') for prompt in prompts]
        # scores_shadow_art = metric_dict['fn'](shadow_art_paths, prompts_shadow_art, model_name=metric_dict['model_name'])
        save_dict = {}
        for i, (image_name, score, score_processed) in enumerate(zip(image_names, scores, scores_processed)):
            save_dict[image_name] = {
                'line_drawing_score': score.item(),
                'line_drawing_processed_score': score_processed.item(),
                # 'shadow_art_score': scores_shadow_art[i].item(),
            }
        with open(save_path, 'w') as f:
            json.dump(save_dict, f)

    return

def get_shadow_scores(output_root, clip_model_name, metric_save_name, dirname='reverse', save_name='outline'): # or shadow
    image_names = os.listdir(os.path.join(output_root, dirname))
    prompts = [open(os.path.join(output_root, 'prompt', image_name.replace('.png', '.txt'))).readlines()[0].strip() for image_name in image_names]
    image_paths = [os.path.join(output_root, dirname, image_name) for image_name in image_names]
    scores = get_clip_scores(image_paths, prompts, model_name=clip_model_name, truncate_prompt=False, keywords_process=True)
    save_path = os.path.join(output_root, f'{metric_save_name}_{save_name}_scores.json')
    save_dict = {}
    for image_name, score in zip(image_names, scores):
        save_dict[image_name] = score.item()
    with open(save_path, 'w') as f:
        json.dump(save_dict, f)

def rank_images(output_root, use_vqa=True, save_topk=4):
    REMOVE_KEYWORDS = ['hand', 'finger']
    image_names = os.listdir(os.path.join(output_root, 'line_drawing_processed'))
    image_reward_scores = json.load(open(os.path.join(output_root, 'imagereward_scores.json')))
    clip_scores = json.load(open(os.path.join(output_root, 'clip_scores.json')))
    hps_scores = json.load(open(os.path.join(output_root, 'hps_scores.json')))

    vqa_score_path = os.path.join(output_root, 'vqa_score_stroke_composition.json') if use_vqa else None
    vqa_scores_stroke = json.load(open(vqa_score_path)) if use_vqa else None
    
    entries = []
    for image_name in image_names:
        ld = clip_scores.get(image_name,{}).get('line_drawing_score')
        pld = clip_scores.get(image_name,{}).get('line_drawing_processed_score')
        ldIR = image_reward_scores.get(image_name,{}).get('line_drawing_score')
        pldIR = image_reward_scores.get(image_name,{}).get('line_drawing_processed_score')
        ldhps = hps_scores.get(image_name,{}).get('line_drawing_score')
        pldhps = hps_scores.get(image_name,{}).get('line_drawing_processed_score')
        prompt = open(os.path.join(output_root, 'prompt', image_name.replace('.png', '.txt'))).read()

        vqa_score_stroke = vqa_scores_stroke.get(image_name,{})['yes'] if use_vqa else None

        entries.append({
            'image_name': image_name,
            'ld': ld,
            'ldIR': ldIR,
            'pld': pld,
            'pldIR': pldIR,
            'ldhps': ldhps,
            'pldhps': pldhps,
            'hps_good': ldhps > pldhps,
            'ir_good': ldIR > pldIR,
            'vqa_score_stroke': vqa_score_stroke,
            'keyword_filter': any(k in prompt.split('\n')[0].split('.')[0].split(' ') for k in REMOVE_KEYWORDS)
        })

    for e in entries:
        def cdf(x):
            return 1/2 * (1 + erf(x/np.sqrt(2)))
        e['clip_score'] = e['ld'] ** 2 / e['pld'] ** 2
        e['ir_score'] = cdf(e['ldIR']) ** 2 - cdf(e['pldIR']) ** 2
        e['hps_score'] = e['ldhps'] ** 2 - e['pldhps'] ** 2

    entries.sort(key=lambda e: (
                                use_vqa and (e['vqa_score_stroke'] < 0.5),
                                e['keyword_filter'],
                                not e['hps_good'] or not e['ir_good'],
                                -e['clip_score'] * e['ir_score'] * e['hps_score']
                                ))
    
    # save the ranking
    save_path = os.path.join(output_root, 'ranking.json')
    with open(save_path, 'w') as f:
        json.dump(entries, f)

    # create a folder called filtered_images
    filtered_images_root = os.path.join(output_root, 'filtered_images')
    os.makedirs(filtered_images_root, exist_ok=True)

    # copy the topk images to the filtered_images folder
    for i in range(save_topk):
        old_path = os.path.join(output_root, "shadow_art", entries[i]["image_name"])
        new_path = os.path.join(filtered_images_root, f'{i:02d}_{entries[i]["image_name"]}')
        os.system(f'cp {old_path} {new_path}')
    
    return

parser = argparse.ArgumentParser()

parser.add_argument('--render_resolution', type=int, default=512)
parser.add_argument('--prompt_detail', type=str, default='high', choices=['low', 'high', 'auto'])
parser.add_argument('--generation_resolution', type=int, default=1024)
parser.add_argument('--dilate_size', type=int, default=4)
parser.add_argument('--system_prompt_path', type=str, default='system_prompts/three_step_0701.txt')
parser.add_argument('--user_prompt_path', type=str, default='system_prompts/three_step_0701_user.txt')
parser.add_argument('--user_character', type=str, default=None)
parser.add_argument('--stroke_image_dir', type=str, default='reverse')
parser.add_argument('--override_prompt', type=str, default=None, help='Use this prompt for all images instead of calling VLM for prompt generation')
parser.add_argument('--min_shadow_region_size', type=int, default=2500)
parser.add_argument('--max_shadow_hole_size', type=int, default=500)
parser.add_argument('--mask_width', type=int, default=3, help='The width of the mask to extract the shadow art')
parser.add_argument('--process_kernel_size', type=int, default=3, help='The kernel size to process the shadow art image')
parser.add_argument('--min_sketch_region_size', type=int, default=40, help='The minimum size of the sketch region to be kept, used for remove small un-removed noise')
parser.add_argument('--erode_pixels', type=int, default=7, help='The number of pixels to erode the object mask')

# rendering parameters
parser.add_argument('--blender_path', type=str, default='blender-4.3.2-linux-x64/blender')
parser.add_argument('--object_scale', type=float, default=0.8)
parser.add_argument('--translate_ratio', type=float, default=0.8)
parser.add_argument('--shadow_length_min', type=float, default=0.8)
parser.add_argument('--shadow_length_max', type=float, default=1.3)
parser.add_argument('--use_gravity', action='store_true', help='Use gravity to position the object')
parser.add_argument('--internal_rotation', action='store_true', help='Enable internal rotation of the object')
parser.add_argument('--sample_distribution', action='store_true', help='Sample the distribution of the elevation')
parser.add_argument('--optimize_object_params', action='store_true', help='Optimize the object parameters')
parser.add_argument('--num_iterations', type=int, default=200, help='Number of iterations for the optimization')
parser.add_argument('--render_count', type=int, default=60, help='Number of renders for each object to fit the distribution')
parser.add_argument('--azimuth_count', type=int, default=12)
parser.add_argument('--elevation_count', type=int, default=4)
parser.add_argument('--multi_object_config', type=str, default=None, choices=['in_out', 'side_by_side', 'up_down'])
parser.add_argument('--do_vqa_score', action='store_true', help='Do VQA score')
parser.add_argument('--do_rank', action='store_true', help='Do rank')
parser.add_argument('--save_topk', type=int, default=4, help='Save the topk images')

# diffusion parameters
parser.add_argument('--guidance_scale', type=float, default=30)
parser.add_argument('--num_inference_steps', type=int, default=28)
parser.add_argument('--lora_path', type=str, required=True)
parser.add_argument('--rank', type=int, default=64)
parser.add_argument('--pretrained_model_name_or_path', type=str, default='black-forest-labs/FLUX.1-Canny-dev')
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--condition_type', type=str, default='canny', choices=['object_shadow', 'canny'])
parser.add_argument('--generation_type', type=str, default='full', choices=['full', 'partial'])

# score parameters
parser.add_argument('--clip_model_name', type=str, default='openai/clip-vit-large-patch14')
parser.add_argument('--imagereward_model_name', type=str, default='ImageReward-v1.0')
parser.add_argument('--hps_model_name', type=str, default='v2.1')
parser.add_argument('--shadow_scores_save_name', type=str, default='clip')

# others
parser.add_argument('--object_filepath', type=str, nargs='+', required=True)
parser.add_argument('--combine_multiple_objects', action='store_true', help='Combine multiple objects into a single object')
parser.add_argument('--output_root', type=str, required=True)
parser.add_argument('--experiment_name', type=str, default=None)


args = parser.parse_args()
os.makedirs(args.output_root, exist_ok=True)

assert not args.sample_distribution or args.internal_rotation, "Internal rotation must be enabled when sample distribution is enabled"
assert not args.optimize_object_params or args.internal_rotation, "Internal rotation must be enabled when optimizing object parameters"
assert not (args.sample_distribution and args.optimize_object_params), "Sample distribution and optimizing object parameters cannot be enabled at the same time"
if len(args.object_filepath) > 1:
    assert args.use_gravity, "with multiple objects, gravity must be enabled"   
assert len(args.object_filepath) <= 5, "Only support up to 5 object files"

generation_resolution = args.generation_resolution

object_filepaths = []
for object_filepath in args.object_filepath:
    if os.path.isdir(object_filepath):
        obj_files = [f for f in os.listdir(object_filepath) if f.endswith('.obj')]
        assert len(obj_files) == 1, f"Multiple .obj files found in {object_filepath}"
        object_filepaths.append(os.path.join(object_filepath, obj_files[0]))
    else:
        object_filepaths.append(object_filepath)

if args.experiment_name is None:
    args.experiment_name = '_and_'.join([os.path.basename(object_filepath).split('.')[0] for object_filepath in object_filepaths])
global_root = os.path.join(args.output_root, args.experiment_name)
os.makedirs(global_root, exist_ok=True)

print(f'Output root: {global_root}', f'Experiment name: {args.experiment_name}')

def main():

    global object_filepaths

    if args.use_gravity:
        animate_object(global_root, object_filepaths, args.blender_path, initial_height=0.1, simulation_frames=250, try_put_down=args.use_gravity, object_scale=args.object_scale, combine_multiple_objects=args.combine_multiple_objects)
        if len(object_filepaths) > 1 and args.combine_multiple_objects: # if multiple objects, use the combined object file
            object_filepaths = [os.path.join(global_root, 'combined_objects.obj')]
            args.use_gravity = False

    if args.sample_distribution or args.optimize_object_params:
        compute_light_elevations(global_root, object_filepaths, args.elevation_count, args.object_scale, args.translate_ratio, args.shadow_length_min, args.shadow_length_max, args.use_gravity, args.blender_path, try_put_down=args.use_gravity)

    if args.sample_distribution:
        compute_irregularity_for_sampling(global_root, object_filepaths, args.render_count, args.object_scale, args.translate_ratio, args.use_gravity)

    if args.optimize_object_params:
        optimize_object_params(global_root, object_filepaths, args.azimuth_count, args.object_scale, args.translate_ratio, args.num_iterations, args.use_gravity)

    print('Rendering images')
    render_image(global_root, object_filepaths, args.azimuth_count, args.elevation_count, args.object_scale, args.translate_ratio, args.render_resolution, 
                 args.blender_path, args.shadow_length_min, args.shadow_length_max, read_scene_params=args.optimize_object_params,
                 synthetic_rotation=False, multi_object_config=args.multi_object_config, internal_rotation=args.internal_rotation, 
                 sample_distribution=args.sample_distribution, use_gravity=args.use_gravity, try_put_down=args.use_gravity)

    print('Processing rendering')
    process_rendering(global_root, args.generation_resolution, args.dilate_size, args.max_shadow_hole_size, args.min_shadow_region_size, 
                      multi_object_config=args.multi_object_config)

    print('Generating prompts')
    prompt_resolution = 512 if args.prompt_detail == 'low' else 768
    if args.user_character:
        system_prompt = open(args.user_prompt_path).read().replace('[user_character]', args.user_character)
    else:
        system_prompt = open(args.system_prompt_path).read()
    create_prompt_gpt(global_root, system_prompt, prompt_resolution, stroke_image_dir=args.stroke_image_dir, override_prompt=args.override_prompt)

    print('Generating line drawings')
    create_line_drawing(global_root, args.guidance_scale, args.num_inference_steps, args.generation_resolution, args.erode_pixels,
                        args.lora_path, args.pretrained_model_name_or_path, args.condition_type, args.batch_size, args.rank)
    
    print('Processing line drawings')
    if args.generation_type == 'partial':
        process_line_drawing_object_shadow(global_root, args.mask_width)
    else:
        process_line_drawing(global_root, args.mask_width, args.process_kernel_size, args.min_sketch_region_size)

    print('Rendering shadow art')
    render_shadow_art(global_root, object_filepaths, args.elevation_count, args.azimuth_count, args.object_scale, args.translate_ratio, args.render_resolution, 
                       args.blender_path, args.shadow_length_min, args.shadow_length_max, args.multi_object_config, args.internal_rotation, 
                       args.use_gravity, try_put_down=args.use_gravity)

    print('Getting scores')
    get_scores(global_root, metrics_list=[
                                            {'metric_save_name': 'clip', 'model_name': args.clip_model_name, 'fn': get_clip_scores},
                                            {'metric_save_name': 'imagereward', 'model_name': args.imagereward_model_name, 'fn': get_imagereward_scores},
                                            {'metric_save_name': 'hps', 'model_name': args.hps_model_name, 'fn': get_hps_scores}
                                         ])

    if args.do_vqa_score:
        print('Computing VQA scores')
        vqa_score_path = os.path.join(global_root, 'vqa_score_stroke_composition.json')
        if os.path.exists(vqa_score_path):
            print(f'VQA scores already exist at {vqa_score_path}')
        else:
            cmd = [
                'python', 'src/vqa_score.py', 
                global_root, 
                '--question', 'stroke composition',
                '--one_layer'
            ]
            subprocess.run(cmd)

    if args.do_rank:
        print('Ranking images')
        rank_images(global_root, use_vqa=args.do_vqa_score, save_topk=args.save_topk)

if __name__ == '__main__':
    main()  
