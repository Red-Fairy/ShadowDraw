import os
from PIL import Image
import cv2
import sys
sys.path.append('.')
from src import encode_image
from openai import OpenAI
import argparse
import math
import json
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import subprocess

SYSTEM_PROMPT = open('system_prompts/vqa.txt', 'r').read()

def get_vqa_score_gpt(image_paths: list[str], questions: list[str], system_prompt: str = None, resolution: int = 512, binaryize: bool = False):

    logprob_dict = {}
    client = OpenAI()

    pbar = tqdm(total=len(image_paths))
    for image_path, question in zip(image_paths, questions):

        image = Image.open(image_path)
        image_base64 = encode_image(image, size=(resolution, resolution), resize_method=Image.Resampling.NEAREST, binaryize=binaryize)

        messages = []
        if system_prompt is not None:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_base64}",
                        "detail": "high"
                    }
                }
            ]
        })

        response = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=messages,
                    logprobs=True,
                    top_logprobs=2,
                )

        logprobs = response.choices[0].logprobs.content[0].top_logprobs

        # accumulate probability for 'yes' and 'no'
        prob_yes, prob_no = 0, 0
        for logprob in logprobs:
            if 'yes' in logprob.token.lower():
                prob_yes += math.exp(logprob.logprob)
            elif 'no' in logprob.token.lower():
                prob_no += math.exp(logprob.logprob)
        logprob_dict[os.path.basename(image_path)] = {'yes': prob_yes, 'no': prob_no}

        pbar.update(1)

    return logprob_dict

def stroke_question(prompt, add_instruction=None):
    processed_prompt = prompt.split('\n')[0].split('.')[0].replace("The provided contour shows an outline of", "Does the highlighted stroke outline") + "?"
    # + "and serve as the core expressive element?" + \
        # " Answer no if the stroke is showing non-expressive body parts such as hands or feet, or accessories such as capes or bags."
    if add_instruction is not None:
        processed_prompt += ' ' + add_instruction
    return processed_prompt

def subject_question(prompt, add_instruction=None):
    image_desc: str = prompt.split('\n')[-1].split('.')[0]
    processed_prompt = 'Does this image show "' + image_desc.replace('A minimalist line drawing of ', '') + '"?'
    if add_instruction is not None:
        processed_prompt += ' ' + add_instruction
    return processed_prompt

def completeness_question(prompt, add_instruction=None):
    processed_prompt = 'Is the outline of' + prompt.split('\n')[0].split('.')[0].replace('The provided contour shows an outline of', '').replace("of a", "of the") + ' complete?'
    if add_instruction is not None:
        processed_prompt += ' ' + add_instruction
    return processed_prompt

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('root_dir', type=str)
    parser.add_argument('--question', type=str, nargs='+', default=['stroke composition', 'subject', 'subject_processed', 'completeness', 'shadow art', 'shadow art object'])
    parser.add_argument('--model', type=str, default='gpt-4.1', choices=['gpt-4.1', 'clip-flant5-xxl', 
        'Qwen/Qwen2.5-VL-7B-Instruct', 'Qwen/Qwen2.5-VL-32B-Instruct', 'Qwen/Qwen2.5-VL-72B-Instruct'])
    parser.add_argument('--one_layer', action='store_true')

    args = parser.parse_args()

    if 'qwen' in args.model.lower():
        args.save_suffix = '_qwen' + args.model.split('VL-')[-1].split('-')[0].lower()
    elif 'flant5' in args.model.lower():
        args.save_suffix = '_flant5'
    else:
        args.save_suffix = ''

    questions = {'stroke composition': {'fn': stroke_question, 'system_prompt': SYSTEM_PROMPT, 'binaryize': False,
                                        'image_subdir': 'line_drawing_overlap', 'save_name': 'vqa_score_stroke_composition' + args.save_suffix}, 
                            'subject': {'fn': subject_question, 'system_prompt': SYSTEM_PROMPT, 'binaryize': True,
                                        'image_subdir': 'line_drawing', 'save_name': 'vqa_score_subject_description' + args.save_suffix}, 
                  'subject_processed': {'fn': subject_question, 'system_prompt': SYSTEM_PROMPT, 'binaryize': True,
                                        'image_subdir': 'line_drawing_processed', 'save_name': 'vqa_score_subject_description_processed' + args.save_suffix}, 
                       'completeness': {'fn': completeness_question, 'system_prompt': SYSTEM_PROMPT, 'binaryize': False,
                                        'image_subdir': 'line_drawing_processed', 'save_name': 'vqa_score_completeness' + args.save_suffix},
                                        }

    questions = {question: questions[question] for question in args.question}

    subdirs = [os.path.join(args.root_dir, x) for x in os.listdir(args.root_dir) if os.path.isdir(os.path.join(args.root_dir, x))] if not args.one_layer else [args.root_dir]
    subdirs = [x for x in subdirs if os.path.exists(os.path.join(x, 'line_drawing'))]

    print(f'Begin processing questions [{", ".join(list(questions.keys()))}] for {len(subdirs)} subdirectories...')

    for i, subdir in enumerate(subdirs):
        print(f'[{i}/{len(subdirs)}] Processing {subdir}...')
        prompt_root = os.path.join(subdir, 'prompt')

        image_names = [x for x in os.listdir(prompt_root) if x.endswith('.txt')]
        prompt_paths = [os.path.join(prompt_root, image_name) for image_name in image_names]
        raw_prompts = [open(prompt_path, 'r').read() for prompt_path in prompt_paths]

        if 'stroke composition' in args.question:
            image_dir = os.path.join(subdir, 'line_drawing_overlap')
            if not os.path.exists(image_dir) or len(os.listdir(image_dir)) != len(image_names):
                print(f'{image_dir} does not exist, generating...')
                subprocess.run(['python', 'src/overlap_contour.py', subdir])
            image_paths = [os.path.join(image_dir, image_name.replace('.txt', '.png')) for image_name in image_names]

        if 'subject' in args.question and questions['subject']['binaryize']:
            image_dir = os.path.join(subdir, 'line_drawing_binary')
            if not os.path.exists(image_dir) or len(os.listdir(image_dir)) != len(image_names):
                print(f'{image_dir} does not exist, generating...')
                subprocess.run(['python', 'helper_scripts/binaryize.py', subdir])
            image_paths = [os.path.join(image_dir, image_name.replace('.txt', '.png')) for image_name in image_names]
        
        for question_name, question_info in questions.items():

            save_path = os.path.join(subdir, question_info['save_name'] + '.json')
            if os.path.exists(save_path):
                print(f'{save_path} already exists, skipping...')
                continue

            print(f'Processing {question_name}...')

            image_paths = [os.path.join(subdir, question_info['image_subdir'], image_name.replace('.txt', '.png')) for image_name in image_names]
            prompts = [question_info['fn'](prompt, add_instruction="Please answer yes or no." if 'clip' in args.model.lower() else None) for prompt in raw_prompts]
            logprob_dict = get_vqa_score_gpt(image_paths=image_paths, questions=prompts, system_prompt=question_info['system_prompt'], binaryize=question_info['binaryize'])

            with open(save_path, 'w') as f:
                json.dump(logprob_dict, f)
