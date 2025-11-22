<h2 align="center">
  <b>ShadowDraw: From Any Object to Shadow–Drawing Compositional Art</b>
</h2>

<div align="center">
    <a href="https://arxiv.org/abs/2504.07940" target="_blank">
    <img src="https://img.shields.io/badge/Paper-arXiv 2025-red" alt="Paper"/></a>
    <a href="https://red-fairy.github.io/ShadowDraw/" target="_blank">
    <img src="https://img.shields.io/badge/Demo-Project Page-blue" alt="Project Page"/></a>
</div>

---

This is the official repository for the paper:  
**ShadowDraw: From Any Object to Shadow–Drawing Compositional Art**

**Authors:** Rundong Luo, Noah Snavely, Wei-Chiu Ma (Cornell University)

---

## 🔧 Installation

First, create a conda environment and install the dependencies:

```bash
git clone https://github.com/red-fairy/ShadowDraw.git
cd ShadowDraw
conda env create -f environment.yml
```

Then, install Blender 4.3.2 following the instructions [here](https://download.blender.org/release/Blender4.3/), unzip it, and specify the path to the Blender executable in the `blender_path` argument of the `main.py` script.

## 🎨 Generate Your Own Shadow-Drawing Art

### Shadow-Drawing Art from a Single Object

First, download the LoRA weights for the line drawing generation model from HuggingFace:
```bash
huggingface-cli download RedFairy/Flux-ShadowDraw-LoRA --local-dir ./checkpoints
```

Then, prepare your 3D object file in `.obj`, `.glb`, or `.ply` format, and run the following command:

```bash
python scripts/launch.py --object_filepaths PATH_TO_OBJECT_FILE \
    --save_name SAVE_NAME \
    --output_root OUTPUT_ROOT \
    --optimize_object_params
```

To accelerate the generation process, you can remove `--optimize_object_params` and instead add `--sample_distribution`. This will fix the azimuths and fit a distribution of object internal rotations with respect to fractal dimension and sample internal rotations accordingly. 

To add gravity to the object, add `--use_gravity`. If specified, the object will be first animated by gravity to find a stable pose and then rendered. 

### Shadow-Drawing Art from Multiple Objects

For multi-object compositions, you can use the following command:

```bash
python scripts/launch.py --object_filepaths PATH_TO_OBJECT_FILES \
    --save_name SAVE_NAME \
    --output_root OUTPUT_ROOT \
    --sample_distribution
```

Here, `PATH_TO_OBJECT_FILES` should be a space-separated list of `.obj`, `.glb`, or `.ply` files.

For user-specified subject, add `--user_character CHARACTER_NAME` and `--system_prompt_path system_prompts/prompt_proposal_user.txt`.


## 📊 Citation
If you find this work useful, please consider citing:

```bibtex
@article{luo2025shadowdraw,
  title={ShadowDraw: From Any Object to Shadow–Drawing Compositional Art},
  author={Luo, Rundong and Snavely, Noah and Ma, Wei-Chiu},
  journal={arXiv preprint arXiv:2504.07940},
  year={2025}   
}
```

The shadow-drawing concept is inspired by the work of Belgian artist Vincent Bal, whose playful works reveal how the cast shadows of everyday objects can seamlessly complete drawn elements. Please refer to his [Instagram page](https://www.instagram.com/vincent_bal/) for more fantastic results.