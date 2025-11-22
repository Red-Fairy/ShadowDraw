import torch
import numpy as np
import torch.nn as nn
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    PointLights,
    DirectionalLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    BlendParams,
    SoftPhongShader,
    HardGouraudShader,
)
from pytorch3d.structures import Meshes
from pytorch3d.transforms import RotateAxisAngle, Transform3d
from matplotlib import pyplot as plt
import sys
import cv2
from PIL import Image
import os
from torch.nn import functional as F
from pytorch3d.io import load_objs_as_meshes, save_obj
import argparse
import math
sys.path.append(".")
from transformers import CLIPModel, AutoTokenizer, AutoProcessor, CLIPImageProcessor
from tqdm import tqdm
from transformers.image_utils import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from torchvision.transforms.functional import resize
from torch.utils.tensorboard import SummaryWriter
import json
import subprocess
from typing import Optional, Tuple
import random   

BLENDER_PATH = '/home/rl897/art-from-phys/blender-4.3.2-linux-x64/blender'

def differentiable_fractal_dimension(binary_image):
    """Uses a weighted sum that approximates 'any pixel occupied' behavior"""
    log_scales = []
    log_counts = []
    
    max_scale = min(binary_image.shape)
    scales = [2 ** i for i in range(0, int(np.log2(max_scale)) + 1)]
    
    if type(binary_image) == np.ndarray:
        binary_image = torch.from_numpy(binary_image).float()
    
    if binary_image.max() > 1:
        binary_image = binary_image / 255.0

    num_filled = None
    
    for scale in scales:
        # Average pooling
        box_sums = F.avg_pool2d(
            binary_image.unsqueeze(0).unsqueeze(0),
            kernel_size=scale,
            stride=scale
        ).squeeze()
        
        n_pixels = scale * scale
        prob_any_pixel = 1 - torch.pow(1 - box_sums, n_pixels)
        
        count = torch.sum(prob_any_pixel)
        
        log_scales.append(math.log(scale))
        log_counts.append(torch.log(count + 1e-8))

        if scale == 1:
            num_filled = count
    
    # Linear regression
    log_scales = torch.tensor(log_scales, device=binary_image.device)
    log_counts = torch.stack(log_counts)
    
    A = torch.stack([log_scales, torch.ones_like(log_scales)], dim=1)
    slope, _ = torch.linalg.lstsq(A[1:], log_counts[1:]).solution
    
    dimension = -slope
    return dimension, num_filled

def load_meshes(object_filepath, output_root, device, scale_target=1, above_plane=False, use_gravity=False, try_put_down=True):

    if object_filepath.endswith('.glb'):
        subprocess.run([BLENDER_PATH, '--background', '--python', 'src/convert_obj.py', '--', '--object_path', object_filepath])
        object_filepath = object_filepath.replace('.glb', '.obj')
    elif object_filepath.endswith('.ply'):
        subprocess.run([BLENDER_PATH, '--background', '--python', 'src/convert_obj.py', '--', '--object_path', object_filepath])
        object_filepath = object_filepath.replace('.ply', '.obj')

    mesh = load_objs_as_meshes([object_filepath], device=device)

    verts = mesh.verts_packed()
    N = verts.shape[0]

    xyz_max = verts.max(0)[0]
    xyz_min = verts.min(0)[0]
    center = (xyz_max + xyz_min) / 2
    if above_plane:
        center[1] = xyz_min[1]
    scale = (xyz_max - xyz_min).max()

    mesh.offset_verts_(-center)
    mesh.scale_verts_((scale_target / float(scale)))

    # rotate the object by xyz (90, 0, 0) to align with blender convention
    tf = Transform3d(device=device).rotate_axis_angle(angle=np.pi / 2, axis='X', degrees=False)
    verts_rotated = tf.transform_points(mesh.verts_padded()) # (num_verts_per_mesh, 3)
    mesh = mesh.update_padded(new_verts_padded=verts_rotated)

    update_bbox = False
    xyz_min, xyz_max = mesh.verts_packed().min(0)[0], mesh.verts_packed().max(0)[0]
    dist_x, dist_y, dist_z = xyz_max - xyz_min
    if try_put_down and (dist_z > 2 * dist_y or dist_z > 2 * dist_x):
        if dist_y > dist_x:
            tf = Transform3d(device=device).rotate_axis_angle(angle=np.pi / 2, axis='Y', degrees=False)
        else:
            tf = Transform3d(device=device).rotate_axis_angle(angle=np.pi / 2, axis='X', degrees=False)
        verts_rotated = tf.transform_points(mesh.verts_padded()) # (num_verts_per_mesh, 3)
        mesh = mesh.update_padded(new_verts_padded=verts_rotated)
        update_bbox = True

    if use_gravity:
        gravity_info = json.load(open(os.path.join(output_root, f'animate_0.json')))
        x_, y_, z_ = gravity_info['rotation_radians']['x'], gravity_info['rotation_radians']['y'], gravity_info['rotation_radians']['z']
        tf = Transform3d(device=device).rotate_axis_angle(angle=x_, axis='X', degrees=False).rotate_axis_angle(angle=y_, axis='Y', degrees=False).rotate_axis_angle(angle=z_, axis='Z', degrees=False)
        verts_rotated = tf.transform_points(mesh.verts_padded()) # (num_verts_per_mesh, 3)
        mesh = mesh.update_padded(new_verts_padded=verts_rotated)
        update_bbox = True

    # rotate the object by xyz (-90, 0, 0) to put the object back to the original orientation
    tf = Transform3d(device=device).rotate_axis_angle(angle=-np.pi / 2, axis='X', degrees=False)
    verts_rotated = tf.transform_points(mesh.verts_padded()) # (num_verts_per_mesh, 3)
    mesh = mesh.update_padded(new_verts_padded=verts_rotated)
    
    if update_bbox:
        verts_new = mesh.verts_packed()
        xyz_max = verts_new.max(0)[0]
        xyz_min = verts_new.min(0)[0]
        center = (xyz_max + xyz_min) / 2
        if above_plane:
            center[1] = xyz_min[1]
        mesh.offset_verts_(-center)

    return mesh

class SceneParam(nn.Module):
    def __init__(self, light_elevation, object_self_rotation, translate_ratio):
        super().__init__()
        # light parameters, in radians
        self.light_azimuth = torch.tensor(0, dtype=torch.float32)
        self.light_elevation_base = torch.tensor(math.radians(light_elevation), dtype=torch.float32)
        self.light_elevation_factor = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.light_elevation_range = 0.05

        # object parameters, in radians
        self.object_azimuth = torch.tensor(0, dtype=torch.float32)
        self.object_self_rotation = nn.Parameter(torch.tensor(math.radians(object_self_rotation), dtype=torch.float32))

        # fixed parameters
        self.translate_ratio = translate_ratio

    @property
    def light_elevation(self):
        return (1 + torch.tanh(self.light_elevation_factor) * self.light_elevation_range) * self.light_elevation_base

    def render_warpped_silhouette(self, mesh: Meshes, image_size=512, 
                                light_distance=10, canvas_radius=2/math.sqrt(3),
                                faces_per_pixel=1, blur_radius_sigma=0, bin_size=None) -> torch.Tensor:

        device = mesh.device
        fov = np.arcsin(canvas_radius / light_distance)

        eye = torch.stack([
            light_distance * torch.cos(self.light_azimuth) * torch.cos(self.light_elevation), 
            light_distance * torch.sin(self.light_elevation), 
            -light_distance * torch.sin(self.light_azimuth) * torch.cos(self.light_elevation)
        ], dim=-1).unsqueeze(0) # (1, 3)

        light_R, light_T = look_at_view_transform(
            eye=eye,
            at=torch.zeros(1, 3), 
            up=torch.tensor([[0, 1, 0]], dtype=eye.dtype, device=eye.device)
        )

        # Create cameras for the light view (to render silhouette)
        light_cameras = FoVPerspectiveCameras(
            device=device,
            R=light_R,
            T=light_T,
            fov=fov,
            degrees=False,
        )

        # Rasterization settings for silhouette rendering
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * blur_radius_sigma,
            faces_per_pixel=faces_per_pixel,
            bin_size=bin_size,
            perspective_correct=False
        )

        # Silhouette renderer from light's perspective
        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=light_cameras,
                raster_settings=raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=blur_radius_sigma))
        )

        # Render silhouette from light's perspective
        silhouette_image = silhouette_renderer(mesh, cameras=light_cameras)[..., -1] # (num_cameras, image_size, image_size, 4)

        # warp this silhouette to the viewing camera's perspective
        # First, we create a grid of points representing the plane at z=0
        xs = torch.linspace(canvas_radius*(-1+1/image_size), canvas_radius*(1-1/image_size), image_size, device=device)
        zs = torch.linspace(canvas_radius*(-1+1/image_size), canvas_radius*(1-1/image_size), image_size, device=device)
        grid_z, grid_x = torch.meshgrid(zs, xs)
        ones = torch.ones_like(grid_x)

        grid_y = mesh.verts_packed()[:, 1].min() * ones

        world_points = torch.stack([grid_x, grid_y, grid_z, ones], dim=-1) # (image_size, image_size, 4)
        world_points = world_points.view(-1, 4) # (image_size*image_size, 4)

        light_ndc_points = light_cameras.transform_points_ndc(world_points[..., :3], image_size=(image_size, image_size)) # (num_cameras, image_size*image_size, 2)

        # interpolate to get the silhouette image
        silhouette_image_warpped = F.grid_sample(silhouette_image.unsqueeze(1),
                                                -light_ndc_points[..., :2].view(1, image_size, image_size, 2), 
                                                mode='bilinear', align_corners=False) # (num_cameras, image_size*image_size, 1)

        return silhouette_image_warpped.permute(0, 2, 3, 1) # (num_cameras, image_size, image_size, 1)

    def forward(self, mesh: Meshes, image_size=512, camera_distance=2, light_distance=10, fov=60,
                faces_per_pixel=1, blur_radius_sigma=0, bin_size=None):

        device = mesh.device
        self.light_azimuth = self.light_azimuth.to(device)
        self.object_azimuth = self.object_azimuth.to(device)
        canvas_radius = camera_distance * np.tan(np.radians(fov) / 2)

        mesh_ = mesh.clone()
    
        # rotate the object
        tf = Transform3d(device=device).rotate_axis_angle(angle=self.light_azimuth+self.object_self_rotation, axis='Y', degrees=False)
        verts_rotated = tf.transform_points(mesh_.verts_padded()) # (num_verts_per_mesh, 3)
        mesh_ = mesh_.update_padded(new_verts_padded=verts_rotated)

        # translate the object
        offset = torch.stack([canvas_radius * self.translate_ratio * torch.cos(self.object_azimuth), 
                              torch.zeros_like(self.object_azimuth),
                              -canvas_radius * self.translate_ratio * torch.sin(self.object_azimuth)], dim=-1)

        mesh_.offset_verts_(offset)

        # render the silhouette
        silhouette_image_warpped = self.render_warpped_silhouette(mesh_, image_size=image_size, 
                                                                light_distance=light_distance, canvas_radius=canvas_radius*1.5,
                                                                faces_per_pixel=faces_per_pixel, blur_radius_sigma=blur_radius_sigma, bin_size=bin_size)

        return silhouette_image_warpped

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--object_filepath", type=str)
    parser.add_argument('--output_root', type=str, required=True)

    # scene configuration
    parser.add_argument('--light_elevations', type=float, nargs='+', required=True)
    parser.add_argument('--object_self_rotations', type=float, nargs='+', default=None)
    parser.add_argument('--azimuths_per_elevation', type=int, default=12)
    parser.add_argument('--random_self_rotations', action='store_true')
    
    parser.add_argument('--use_gravity', action='store_true', help='use gravity')
    parser.add_argument('--translate_ratio', type=float, default=0.8)

    # rendering parameters
    parser.add_argument("--fov", type=float, default=60)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--camera_distance", type=float, default=2)
    parser.add_argument("--light_distance", type=float, default=10)
    parser.add_argument("--faces_per_pixel", type=int, default=20)
    parser.add_argument("--blur_radius_sigma", type=float, default=1e-7)
    parser.add_argument('--bin_size', type=int, default=None)
    parser.add_argument('--scale_target', type=float, default=0.8)
    parser.add_argument('--disable_put_down', action='store_true', help='Disable trying to put the object down')

    # optimization parameters
    parser.add_argument('--num_iterations', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--clip_grad_norm', type=float, default=1.0)
    parser.add_argument('--clip_grad_norm_elevation', type=float, default=1.0)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['step', 'cosine'])
    parser.add_argument('--save_interval', type=int, default=50)

    parser.add_argument('--params_to_optimize', type=str, default=['object_self_rotation', 'light_elevation'], nargs='+',
                        choices=['all', 'light', 'object', 'light_elevation', 'object_self_rotation'])

    args = parser.parse_args()

    # --- set the device ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- load the mesh ---
    object_filepath = args.object_filepath
    mesh = load_meshes(object_filepath, output_root=args.output_root, above_plane=True, scale_target=args.scale_target, device=device, use_gravity=args.use_gravity, try_put_down=not args.disable_put_down)

    if args.random_self_rotations:
        object_self_rotations = [360 * random.random() for _ in range(len(args.light_elevations))]
    elif args.object_self_rotations is not None:
        object_self_rotations = args.object_self_rotations
    else:
        assert len(args.light_elevations) % args.azimuths_per_elevation == 0
        object_self_rotations = [360 * i / args.azimuths_per_elevation for i in range(args.azimuths_per_elevation)] * (len(args.light_elevations) // args.azimuths_per_elevation) 

    optimized_params = []

    for light_elevation, object_self_rotation in zip(args.light_elevations, object_self_rotations):

        scene_param = SceneParam(light_elevation=light_elevation, object_self_rotation=object_self_rotation, translate_ratio=args.translate_ratio).to(device)

        params_to_optimize = []
        if args.params_to_optimize == 'all':
            params_to_optimize = scene_param.parameters()
        else:
            for param_name, param in scene_param.named_parameters():
                if any(param_name.startswith(p) for p in args.params_to_optimize):
                    params_to_optimize.append(param)

        for param in params_to_optimize:
            param.requires_grad = True

        optimizer = torch.optim.AdamW(params_to_optimize, lr=args.lr)
        if args.scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_iterations, eta_min=0)
        elif args.scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.num_iterations // 2, gamma=0.1)
        else:
            raise ValueError(f"Invalid scheduler: {args.scheduler}")

        initial_loss, final_loss = 0, 0

        pbar = tqdm(total=args.num_iterations)
        for i in range(args.num_iterations):
            
            silhouette_image_warpped = scene_param(mesh, image_size=args.image_size, 
                                                        camera_distance=args.camera_distance, 
                                                        light_distance=args.light_distance,
                                                        blur_radius_sigma=args.blur_radius_sigma,
                                                        bin_size=args.bin_size,
                                                        faces_per_pixel=args.faces_per_pixel,
                                                        fov=args.fov).squeeze(-1).squeeze(0) # (1, image_size, image_size)

            # --- compute the loss ---
            fractal_dimension, filled_count = differentiable_fractal_dimension(silhouette_image_warpped) # want to maximize the fractal dimension

            filled_ratio = filled_count / args.image_size**2
            loss = -(fractal_dimension + filled_ratio * 5)

            optimizer.zero_grad()
            loss.backward()

            if i == 0:
                initial_loss = loss.item()
            
            if i == args.num_iterations - 1:
                final_loss = loss.item()

            # clip the gradients
            torch.nn.utils.clip_grad_norm_(params_to_optimize, max_norm=args.clip_grad_norm)

            optimizer.step()
            scheduler.step()

            pbar.set_description(f"Loss: {loss.item():.6f}, fractal_dimension: {fractal_dimension:.6f}, filled_ratio: {filled_ratio:.6f}, "
                                f"lr: {scheduler.get_last_lr()[0]:.6f}, "
                                f"object_self_rotation: {math.degrees(scene_param.object_self_rotation.item()):.6f}, "
                                f"light_elevation: {math.degrees(scene_param.light_elevation.item()):.6f}")
            pbar.update(1)

        def make_normalized(angle):
            while angle < 0:
                angle += 360
            while angle > 360:
                angle -= 360
            return angle

        optimized_params.append({
            'light_elevation': math.degrees(scene_param.light_elevation.item()),
            'light_elevation_initial': math.degrees(light_elevation),
            'object_self_rotation': make_normalized(math.degrees(scene_param.object_self_rotation.item())),
            'object_self_rotation_initial': math.degrees(object_self_rotation),
            'fractal_dimension_initial': initial_loss,
            'fractal_dimension_final': final_loss
        })

        print(f"Fractal dimension changed from {initial_loss:.6f} to {final_loss:.6f}")
        print(f"Object self rotation changed from {object_self_rotation:.6f} to {math.degrees(scene_param.object_self_rotation.item()):.6f}")
        print(f"Light elevation changed from {light_elevation:.6f} to {math.degrees(scene_param.light_elevation.item()):.6f}")

    with open(os.path.join(args.output_root, 'object_params_optimized.json'), 'w') as f:
        json.dump(optimized_params, f)


if __name__ == "__main__":
    main()