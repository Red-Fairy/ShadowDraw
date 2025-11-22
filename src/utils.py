from PIL import Image
from typing import List
import io
import base64
import numpy as np
from scipy.ndimage import binary_fill_holes, label

def make_grid(images: List[Image.Image]):
    # arrange M images into (M // 2) * 2 grid, if M is odd, the last image is at the bottom left, and add a white background
    if len(images) % 2 == 1:
        images.append(Image.new('RGB', images[0].size, (255, 255, 255)))

    num_rows = len(images) // 2
    num_cols = 2
    image_size = images[0].size
    grid = Image.new('RGB', (num_cols * image_size[0], num_rows * image_size[1]))

    for i, image in enumerate(images):
        grid.paste(image, (i % num_cols * image_size[0], i // num_cols * image_size[1]))
    return grid

def encode_image(image_or_path: Image.Image | str, size=(512, 512), binaryize=False, resize_method=Image.Resampling.NEAREST):

    buffer = io.BytesIO()
    if isinstance(image_or_path, str):
        with Image.open(image_or_path) as img:
            img = img.convert("RGB")  # Ensure it's in RGB mode
            img = img.resize(size, resize_method)    # Resize image
    else:
        img = image_or_path.convert("RGB").resize(size, resize_method)

    if binaryize:
        img = np.array(img.convert('L'))
        img = np.where(img > 128, 255, 0).astype(np.uint8)
        img = Image.fromarray(img).convert('RGB')

    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def fill_small_holes(binary_shadow_image, max_shadow_hole_size=1000):

    # Fill all holes for reference
    fully_filled = binary_fill_holes(binary_shadow_image)

    # Difference gives holes
    holes = fully_filled - binary_shadow_image

    # Label connected components in the hole mask
    labeled_holes, num_features = label(holes)

    # Fill only small holes
    for i in range(1, num_features + 1):
        component = (labeled_holes == i)
        if np.sum(component) <= max_shadow_hole_size:
            binary_shadow_image[component] = 1

    return binary_shadow_image