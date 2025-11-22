import torch
import numpy as np
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
    HardPhongShader,
    HardGouraudShader,
    TexturesVertex
)
from pytorch3d.structures import Meshes
from matplotlib import pyplot as plt
import sys
import cv2
from PIL import Image
import os
from torch.nn import functional as F
from pytorch3d.io import load_objs_as_meshes, load_obj
import argparse
import math
from scipy.ndimage import label, binary_fill_holes, generate_binary_structure
sys.path.append(".")

def render_warpped_silhouette(mesh, light_elevations, light_azimuths, fov=60, image_size=512, camera_distance=2, 
                              faces_per_pixel=50, blur_radius_sigma=1e-4, bin_size=0, device=torch.device("cuda:0")):

    canvas_size = camera_distance * np.tan(np.radians(fov) / 2) * 2

    num_cameras = len(light_elevations)

    dist_light = torch.tensor([camera_distance] * num_cameras)

    light_R, light_T = look_at_view_transform(
        dist=dist_light,
        elev=light_elevations,   # Light coming from above
        azim=light_azimuths     # Light coming from the front
    )

    # Create cameras for the light view (to render silhouette)
    light_cameras = FoVOrthographicCameras(
        device=device,
        R=light_R,
        T=light_T,
        min_x=-canvas_size/2,
        max_x=canvas_size/2,
        min_y=-canvas_size/2,
        max_y=canvas_size/2,
    )

    # Rasterization settings for silhouette rendering
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * blur_radius_sigma,
        faces_per_pixel=faces_per_pixel,
        bin_size=bin_size
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
    silhouette = silhouette_renderer(mesh.extend(num_cameras), cameras=light_cameras) # (num_cameras, image_size, image_size, 4)

    # The silhouette image is the alpha channel
    silhouette_image = silhouette[..., 3] # (num_cameras, image_size, image_size)

    # Now we need to warp this silhouette to the viewing camera's perspective
    # First, we create a grid of points representing the plane at z=-1
    xs = torch.linspace(-canvas_size/2+canvas_size/2/image_size, canvas_size/2-canvas_size/2/image_size, image_size, device=device)
    zs = torch.linspace(-canvas_size/2+canvas_size/2/image_size, canvas_size/2-canvas_size/2/image_size, image_size, device=device)
    grid_z, grid_x = torch.meshgrid(zs, xs)
    ones = torch.ones_like(grid_x)

    grid_y = mesh.verts_packed()[:, 1].min() * ones

    world_points = torch.stack([grid_x, grid_y, grid_z, ones], dim=-1) # (image_size, image_size, 4)
    world_points = world_points.view(-1, 4) # (image_size*image_size, 4)

    light_ndc_points = light_cameras.transform_points_ndc(world_points[..., :3].unsqueeze(0).repeat(num_cameras, 1, 1), image_size=(image_size, image_size)) # (num_cameras, image_size*image_size, 2)

    # interpolate to get the silhouette image
    silhouette_image_warpped = F.grid_sample(silhouette_image.unsqueeze(1),
                                            -light_ndc_points[..., :2].view(num_cameras, image_size, image_size, 2), 
                                            mode='bilinear', align_corners=False) # (num_cameras, image_size*image_size, 1)

    return silhouette_image_warpped

def render_silhouette_up(mesh, fov=60, image_size=512, camera_distance=2, 
                         faces_per_pixel=50, blur_radius_sigma=1e-4, bin_size=0, device=torch.device("cuda:0")):


    up_camera_R, up_camera_T = look_at_view_transform(
        eye=((0, camera_distance, 0),),
        at=((0, 0, 0),),
        up=((0, 0, -1),)
    )

    up_cameras = FoVPerspectiveCameras(
        device=device,
        R=up_camera_R,
        T=up_camera_T,
        fov=fov
    )

    # render the silhouette of the mesh
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * blur_radius_sigma,
        faces_per_pixel=faces_per_pixel,
        bin_size=bin_size
    )

    # Silhouette renderer from light's perspective
    silhouette_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=up_cameras,
            raster_settings=raster_settings
        ),
        shader=SoftSilhouetteShader(blend_params=BlendParams(sigma=blur_radius_sigma))
    )

    silhouette_image_up = silhouette_renderer(mesh)[..., -1]

    return silhouette_image_up

def render_rgb(mesh, 
               light_elevations, light_azimuths, 
               fov=60, image_size=512, camera_distance=2, 
               faces_per_pixel=50, blur_radius_sigma=1e-4, bin_size=0, device=torch.device("cuda:0")):
    
    num_cameras = len(light_elevations)
    
    up_camera_R, up_camera_T = look_at_view_transform(
        eye=((0, camera_distance, 0),),
        at=((0, 0, 0),),
        up=((0, 0, -1),)
    )

    up_cameras = FoVPerspectiveCameras(
        device=device,
        R=up_camera_R,
        T=up_camera_T,
        fov=fov
    )

    # render the silhouette of the mesh
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=np.log(1. / 1e-4 - 1.) * blur_radius_sigma,
        faces_per_pixel=faces_per_pixel,
        bin_size=bin_size
    )

    dist_light = torch.tensor([camera_distance])

    light_R, light_T = look_at_view_transform(
        dist=dist_light,
        elev=light_elevations,   # Light coming from above
        azim=light_azimuths     # Light coming from the front
    )

    lights = DirectionalLights(direction=light_T, device=device)

    rgb_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=up_cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(
            device=device, 
            cameras=up_cameras,
            lights=lights
        )   
    )

    rgb_images = rgb_renderer(mesh.extend(num_cameras), cameras=up_cameras)

    return rgb_images

def load_meshes(obj_filepath, device=torch.device("cuda:0"), center_mode="middle", 
                scale_target=1, above_plane=False, with_mtl=False):

    '''
    center_mode: mean of verts of center of the bounding box of the mesh
    '''
    if with_mtl:
        mesh = load_objs_as_meshes([obj_filepath], device=device)
    else:
        verts, faces_idx, aux = load_obj(obj_filepath, device=device)
        verts_rgb = torch.ones_like(verts)[None]
        textures = TexturesVertex(verts_features=verts_rgb.to(device))
        mesh = Meshes(verts=[verts], faces=[faces_idx.verts_idx], textures=textures)

    # We scale normalize and center the target mesh to fit in a sphere of radius 1 
    # centered at (0,0,0). (scale, center) will be used to bring the predicted mesh 
    # to its original center and scale.  Note that normalizing the target mesh, 
    # speeds up the optimization but is not necessary!
    verts = mesh.verts_packed()
    N = verts.shape[0]

    if center_mode == "mean":
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0]) * 2
    elif center_mode == "middle":
        xyz_max = verts.max(0)[0]
        xyz_min = verts.min(0)[0]
        center = (xyz_max + xyz_min) / 2
        if above_plane:
            center[1] = xyz_min[1]
        scale = (xyz_max - xyz_min).max()
    else:
        raise ValueError(f"Invalid center_mode: {center_mode}")

    mesh.offset_verts_(-center)

    mesh.scale_verts_((scale_target / float(scale)))

    return mesh