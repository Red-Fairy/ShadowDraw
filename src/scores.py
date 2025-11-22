import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
import sys
sys.path.append('.')
import ImageReward as reward
from typing import List
import re
import os
import hpsv2

def extract_subject_prefix_robust(caption):
    """
    More robust version with multiple extraction strategies
    """
    
    # Strategy 1: Pattern matching with common stop words and punctuation
    stop_words = [
        'with', 'in', 'on', 'at', 'having', 'showing', 'displaying'
    ]
    
    # Add ALL -ing and -ed endings, plus punctuation
    stop_pattern = '|'.join(stop_words + [r'[a-zA-Z]+ing', r'[a-zA-Z]+ed'])
    
    pattern = f'^(A minimalist line drawing of (?:an? )?[a-zA-Z\\s]+?)(?:[,.]|\\s+(?:{stop_pattern})\\b)'
    
    match = re.match(pattern, caption, re.IGNORECASE)
    if match:
        return match.group(1).strip().replace('A minimalist line drawing of ', '')
    
    # Fallback: Count words after "of" and take reasonable number
    prefix_match = re.match(r'^A minimalist line drawing of ((?:an? )?.*)', caption, re.IGNORECASE)
    if prefix_match:
        remaining = prefix_match.group(1)
        words = remaining.split()
        
        # Take first 1-3 words as subject, depending on content
        if len(words) == 1:
            subject_words = words[:1]
        elif len(words) == 2:
            subject_words = words[:2]
        elif len(words) == 3:
            subject_words = words[:3]  # article + adjective + noun
        elif words[0].lower() in ['a', 'an']:
            subject_words = words[:4]  # article + adjective + compound noun
        else:
            subject_words = words[:3]  # adjective + noun or compound noun
        
        return ' '.join(subject_words)
    
    return caption

def process_prompt_keywords(prompt: str, keywords: List[str] = ['resembles', 'outlines', 'forms', 'suggests']):

    prompt = prompt.split('.')[0].split(',')[0] + '.' # keep the first sentence

    # for keyword in keywords:
    #     if keyword in prompt:
    #         prompt = prompt[prompt.find(keyword) + len(keyword) :].strip()
    #         break

    prompt = prompt.replace('An outline of ', 'A silhouette of ')
    prompt = prompt.replace('The provided contour shows an', 'An')

    return prompt

def process_prompt_replace(prompt: str):

    prompt = prompt.split('.')[0] + '.' # keep the first sentence
    # prompt = prompt.replace("A minimalist line drawing of ", "")
    # prompt = prompt[0].upper() + prompt[1:]

    return prompt

def get_clip_scores(image_paths: List[str], prompts: List[str], model_name: str = 'openai/clip-vit-large-patch14', 
                    device: str = 'cuda', binarize: bool = True, batch_size: int = 32, 
                    truncate_prompt: bool = True, keywords_process: bool = False):

    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model = model.to(device)

    scores = torch.zeros(len(image_paths))

    assert len(image_paths) == len(prompts)

    if truncate_prompt:
        # prompts = [extract_subject_prefix_robust(prompt) for prompt in prompts]
        prompts = [process_prompt_replace(prompt) for prompt in prompts]

    if keywords_process:
        prompts = [process_prompt_keywords(prompt) for prompt in prompts]

    # Process images in batches
    for i in range(0, len(image_paths), batch_size):

        batch_files = image_paths[i:i + batch_size if i + batch_size < len(image_paths) else len(image_paths)]
        batch_prompts = prompts[i:i + batch_size if i + batch_size < len(image_paths) else len(image_paths)]
        batch_images = [Image.open(img_path) if img_path is not None else Image.new('RGB', (1024, 1024), (255, 255, 255)) for img_path in batch_files]
        
        if binarize:
            batch_images = [image.convert('L').point(lambda x: 0 if x < 128 else 255, '1').convert('RGB') for image in batch_images]

        # Prepare inputs
        inputs = processor(
            text=batch_prompts,
            images=batch_images,
            return_tensors="pt",
            padding=True
        ).to(device)

        # Get similarity scores
        with torch.no_grad():
            outputs = model(**inputs)
            image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True) # [batch_size, 768]
            text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True) # [batch_size, 768]

            similarity = 100.0 * torch.einsum('id,id->i', image_embeds, text_embeds)

        # Handle both single image and batch cases
        scores[i:i + batch_size] = similarity

    return scores

def get_imagereward_scores(image_paths: List[str], prompts: List[str], device: str = 'cuda', model_name: str = 'ImageReward-v1.0',
                           binarize: bool = True, truncate_prompt: bool = True):

    model = reward.load(model_name).to(device)

    scores = torch.zeros(len(image_paths))
    assert len(image_paths) == len(prompts)

    if truncate_prompt:
        # prompts = [extract_subject_prefix_robust(prompt) for prompt in prompts]
        prompts = [process_prompt_replace(prompt) for prompt in prompts]

    # Process images in batches
    for i, (image_path, prompt) in enumerate(zip(image_paths, prompts)):
        image = Image.open(image_path) if image_path is not None else Image.new('RGB', (1024, 1024), (255, 255, 255))
        if binarize:
            image = image.convert('L').point(lambda x: 0 if x < 128 else 255, '1').convert('RGB')
        score = model.score(prompt, image)
        scores[i] = score

    return scores

def get_hps_scores(image_paths: List[str], prompts: List[str], device: str = 'cuda', model_name: str = 'v2.1',
                   binarize: bool = True, truncate_prompt: bool = True):

    scores = torch.zeros(len(image_paths))

    if truncate_prompt:
        prompts = [process_prompt_replace(prompt) for prompt in prompts]

    for i, (image_path, prompt) in enumerate(zip(image_paths, prompts)):
        image = Image.open(image_path)
        if binarize:
            image = image.convert('L').point(lambda x: 0 if x < 128 else 255, '1').convert('RGB')
        result = hpsv2.score(image, prompt, hps_version=model_name)
        scores[i] = torch.tensor(result, dtype=torch.float32)

    return scores

if __name__ == "__main__":
    prompt_root = '/home/rl897/art-from-phys/gradio_demo/results/real_object_kushal/0701-3step/shadow0.7-1.3_updiff_translate0.8_scale0.8_resize0.8_rank128_ckpt750/LetterA/prompt'

    prompt_paths = [x for x in os.listdir(prompt_root) if x.endswith('.txt')]
    for prompt_path in prompt_paths:
        prompt = open(os.path.join(prompt_root, prompt_path), 'r').readlines()[-1].strip()
        print(extract_subject_prefix_robust(prompt))
    