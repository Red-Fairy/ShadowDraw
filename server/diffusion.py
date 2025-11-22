from diffusers import FluxTransformer2DModel, FluxPipeline, FluxControlInpaintPipeline
from diffusers.utils import convert_unet_state_dict_to_peft
from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
import torch
import io
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import socket
import argparse
import json
import cv2
from PIL import Image, ImageOps
import numpy as np
import os
import traceback

def prepare_model(lora_path, pretrained_model_name_or_path, device, rank):
    
    transformer = FluxTransformer2DModel.from_pretrained(
        pretrained_model_name_or_path, 
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    ).to(device)

    target_modules = [
        "attn.to_k",
        "attn.to_q",
        "attn.to_v",
        "attn.to_out.0",
        "attn.add_k_proj",
        "attn.add_q_proj",
        "attn.add_v_proj",
        "attn.to_add_out",
        "ff.net.0.proj",
        "ff.net.2",
        "ff_context.net.0.proj",
        "ff_context.net.2",
        "norm1_context.linear",
        "norm1.linear",
        "norm.linear",
        "proj_mlp",
        "proj_out",
    ]

    # process the lora state dict
    lora_state_dict = FluxPipeline.lora_state_dict(lora_path)
    transformer_state_dict = {
        f'{k.replace("transformer.module.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.module.")
    }
    if transformer_state_dict == {}:
        transformer_state_dict = {
            f'{k.replace("transformer.", "")}': v for k, v in lora_state_dict.items() if k.startswith("transformer.")
        }
    transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)

    if any('embedder' in k for k in lora_state_dict.keys()):
        target_modules.append('x_embedder')

    use_dora = any('lora_magnitude_vector' in k for k in lora_state_dict.keys())
    transformer_lora_config = LoraConfig(
        use_dora=use_dora,
        r=rank,
        lora_alpha=rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer = inject_adapter_in_model(transformer_lora_config, transformer)
    result = set_peft_model_state_dict(transformer, transformer_state_dict, adapter_name="default")
    
    assert len(result.unexpected_keys) == 0, f"Unexpected keys: {result.unexpected_keys}"
    assert all(not 'lora' in k.lower() for k in result.missing_keys), f"Missing keys: {result.missing_keys}"

    return transformer

def run_diffusion(json_data, pipeline):

    params = json.loads(json_data)

    prompts_or_prompt_filepaths: list[str] = params["prompt"]
    control_image_path_or_root: str = params["control_image_path"]
    mask_image_path_or_root: str = params["mask_image_path"]
    output_paths: list[str] = params["output_paths"]
    guidance_scale = params["guidance_scale"]
    num_inference_steps = params["num_inference_steps"]
    generation_resolution = params["generation_resolution"]

    assert type(prompts_or_prompt_filepaths) == list, "prompt must be a list of strings"
    if os.path.isfile(prompts_or_prompt_filepaths[0]):
        prompt = [open(prompt_filepath).readlines()[-1].strip().split('\t')[-1] for prompt_filepath in prompts_or_prompt_filepaths]
    else:
        prompt = prompts_or_prompt_filepaths

    rotate_degrees = params["rotate_degrees"] if "rotate_degrees" in params else None
    erode_pixels = params["erode_pixels"] if "erode_pixels" in params else 7
    batch_size = params["batch_size"] if "batch_size" in params else 8

    def process_mask(mask_image_path):
        mask = cv2.erode(np.array(Image.open(mask_image_path).split()[-1]), np.ones((erode_pixels, erode_pixels), np.uint8))
        mask = np.where(mask > 128, 255, 0).astype(np.uint8)
        mask = ImageOps.invert(Image.fromarray(mask))
        return mask

    if os.path.isfile(control_image_path_or_root) and os.path.isfile(mask_image_path_or_root):
        canny_edge_image = Image.open(control_image_path_or_root).convert('RGB')
        mask = process_mask(mask_image_path_or_root)

        if rotate_degrees is not None:
            control_images = [canny_edge_image.rotate(rotate_degree, fillcolor=0) for rotate_degree in rotate_degrees]
            masks = [mask.rotate(rotate_degree, fillcolor=255) for rotate_degree in rotate_degrees]
        else:
            control_images = [canny_edge_image]
            masks = [mask]

    elif os.path.isdir(control_image_path_or_root) and os.path.isdir(mask_image_path_or_root):
        image_names = [os.path.basename(output_path) for output_path in output_paths]
        assert rotate_degrees is None, "rotate_degrees must be None when control_image_paths and mask_image_paths are lists"
        control_images = [Image.open(os.path.join(control_image_path_or_root, image_name)).convert('RGB') for image_name in image_names]
        masks = [process_mask(os.path.join(mask_image_path_or_root, image_name)) for image_name in image_names]

    # get the indices of images that output_path already exists
    output_path_indices = []
    for i, output_path in enumerate(output_paths):
        if os.path.exists(output_path):
            output_path_indices.append(i)
    
    output_paths = [output_path for i, output_path in enumerate(output_paths) if i not in output_path_indices]
    control_images = [control_image for i, control_image in enumerate(control_images) if i not in output_path_indices]
    masks = [mask for i, mask in enumerate(masks) if i not in output_path_indices]
    prompt = [prompt for i, prompt in enumerate(prompt) if i not in output_path_indices]

    for i in range(0, len(output_paths), batch_size):

        batch_end = min(i + batch_size, len(output_paths))
        batch_output_paths = output_paths[i:batch_end]
        batch_control_images = control_images[i:batch_end]
        batch_masks = masks[i:batch_end]
        batch_prompts = prompt[i:batch_end]

        generated_images = pipeline(
            prompt=batch_prompts,
            image=torch.ones((batch_end - i, 3, generation_resolution, generation_resolution)),
            control_image=batch_control_images,
            mask_image=batch_masks,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=1,
            height=generation_resolution,
            width=generation_resolution
        ).images

        for j, image in enumerate(generated_images):
            os.makedirs(os.path.dirname(batch_output_paths[j]), exist_ok=True)
            image.save(batch_output_paths[j])

    print(f'Finished generation for {len(output_paths)} images')
    

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_path", type=str)
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="black-forest-labs/FLUX.1-Canny-dev")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--json_data", type=str, required=True)

    args = parser.parse_args()
    device = 'cuda'
    transformer = prepare_model(args.lora_path, args.pretrained_model_name_or_path, device=device, rank=args.rank)

    pipeline = FluxControlInpaintPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to(device)

    run_diffusion(args.json_data, pipeline)

if __name__ == "__main__":
    main()