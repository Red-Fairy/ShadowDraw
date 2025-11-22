import argparse
import json
import math
import os
import random
import sys
from typing import Any, Callable, Dict, List, Generator, Literal, Optional, Tuple

import bpy
import numpy as np
from mathutils import Matrix, Vector
import collections
import traceback
import shutil
import socket
import argparse
from scipy.optimize import fsolve
import bmesh
from mathutils.bvhtree import BVHTree
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import griddata
from pathlib import Path
import tempfile
import subprocess

def images_to_video(image_paths, out_path, fps=10):
    image_paths = [Path(p) for p in image_paths]

    # Ensure consistent order (e.g., by filename) — or keep as-is if you prefer
    image_paths.sort()

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)

        # Copy (or re-encode) to a numbered sequence
        # 1-based indexing is conventional for ffmpeg patterns
        for i, src in enumerate(image_paths, start=1):
            # Keep extension the same; ffmpeg can mix formats, but PNG/JPG is safest
            numbered = tmp / f"frame_{i:06d}{src.suffix.lower()}"
            shutil.copy(src, numbered)

        cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(fps),
            "-i", str(tmp / "frame_%06d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-preset", "medium",
            out_path,
        ]
        subprocess.run(cmd, check=True)

class InverseTransformSampler:
    def __init__(self, x_data, y_data):
        # Sort data by x
        sorted_indices = np.argsort(x_data)
        self.x_sorted = x_data[sorted_indices]
        self.y_sorted = y_data[sorted_indices]
        
        # Normalize to probability
        self.y_normalized = self.y_sorted / np.trapezoid(self.y_sorted, self.x_sorted)
        
        # Compute CDF
        self.cdf_values = np.concatenate([[0], cumulative_trapezoid(self.y_normalized, self.x_sorted)])
        self.cdf_values = self.cdf_values / self.cdf_values[-1]  # Ensure it ends at 1
        
        # Create inverse CDF function
        self.inverse_cdf = interp1d(self.cdf_values, self.x_sorted, 
                                   bounds_error=False, fill_value=(0, 100))
    
    def sample(self, n_samples=1000):
        """Sample using inverse transform method"""
        u = np.random.uniform(0, 1, n_samples)
        return self.inverse_cdf(u)
    
    def pdf(self, x):
        """Approximate PDF using interpolation"""
        pdf_interp = interp1d(self.x_sorted, self.y_normalized, 
                             bounds_error=False, fill_value=0)
        return pdf_interp(x)

    def plot_distribution(self, samples, save_path):

        plt.figure(figsize=(20, 8))
        plt.subplot(1, 2, 1)

        x_smooth = np.linspace(0, 360, 360)
        y_smooth = self.pdf(x_smooth) * np.trapezoid(self.y_normalized, self.x_sorted)

        plt.plot(x_smooth, y_smooth, label='Fitted distribution')
        plt.xlabel('x')
        plt.ylabel('y (probability)')
        plt.legend()
        plt.title('Fitted Distribution')

        plt.subplot(1, 2, 2)
        plt.hist(samples, bins=60, density=True, alpha=0.7, label='Samples')
        plt.xlabel('x')
        plt.ylabel('Probability density')
        plt.title('Sampled x values')
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

class VanillaSampler2D:
    def __init__(self, x_data, y_data):
        self.x_data = x_data # (n, 2)
        self.y_data = y_data # (n,)
        self.y_data = self.y_data / np.sum(self.y_data)

    def sample(self, n_samples=1000):
        rng = np.random.default_rng()
        return rng.choice(self.x_data, size=n_samples, p=self.y_data, replace=True, axis=0) # (n_samples, 2)

    def plot_distribution(self, samples, save_path=None):
        """Plot the estimated 2D distribution"""
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Plot original data
        axes[0].scatter(self.x_data[:, 0], self.x_data[:, 1], 
                        c=self.y_data, cmap='viridis', alpha=0.7)
        axes[0].set_title('Original Data')
        axes[0].set_xlabel('X')
        axes[0].set_ylabel('Y')
        
        # Plot samples
        axes[1].scatter(samples[:, 0], samples[:, 1], alpha=0.5, s=1)
        axes[1].set_title('Generated Samples')
        axes[1].set_xlabel('X')
        axes[1].set_ylabel('Y')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    
IMPORT_FUNCTIONS: Dict[str, Callable] = {
    "obj": bpy.ops.wm.obj_import,
    "glb": bpy.ops.import_scene.gltf,
    "gltf": bpy.ops.import_scene.gltf,
    "usd": bpy.ops.import_scene.usd,
    "fbx": bpy.ops.import_scene.fbx,
    "stl": bpy.ops.import_mesh.stl,
    "usda": bpy.ops.import_scene.usda,
    "dae": bpy.ops.wm.collada_import,
    "ply": bpy.ops.wm.ply_import,
    "abc": bpy.ops.wm.alembic_import,
    "blend": bpy.ops.wm.append,
}

def reset_cameras() -> None:
    """Resets the cameras in the scene to a single default camera."""
    # Delete all existing cameras
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="CAMERA")
    bpy.ops.object.delete()

    # Create a new camera with default properties
    bpy.ops.object.camera_add()

    # Rename the new camera to 'NewDefaultCamera'
    new_camera = bpy.context.active_object
    new_camera.name = "Camera"

    # Set the new camera as the active camera for the scene
    scene.camera = new_camera

def set_camera_look_at_center(
    center_distance: float = 15,
    jitter: float = 0.0,
) -> bpy.types.Object:
    """Sets the camera to look at the center of the scene.

    Returns:
        bpy.types.Object: The camera object.
    """
    camera = bpy.data.objects["Camera"]

    azimuth = 0
    # elevation = random.uniform(0, np.radians(jitter))
    elevation = 0

    # Compute the camera location.
    x = center_distance * np.cos(azimuth) * np.sin(elevation)
    y = center_distance * np.sin(azimuth) * np.sin(elevation)
    z = center_distance * np.cos(elevation)

    camera.location = Vector(np.array([x, y, z]))
    
    # Compute the direction from the camera to the origin.
    direction = -camera.location
    if direction.length == 0:
        raise ValueError("Camera location cannot be at the origin.")
    
    # Compute rotation: point the camera's -Z toward the origin.
    rot_quat = direction.to_track_quat("-Z", "Y")
    
    # Convert to Euler and force roll (the rotation around the view axis) to zero.
    camera.rotation_mode = 'XYZ'
    eul = rot_quat.to_euler('XYZ')
    # Here, the roll is typically the Z component (depending on rotation order).
    eul.z = 0
    camera.rotation_euler = eul

    return camera

def reset_scene() -> None:
    """Resets the scene to a clean state.

    Returns:
        None
    """
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)

    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)

    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a model with a supported file extension into the scene.

    Args:
        object_path (str): Path to the model file.

    Raises:
        ValueError: If the file extension is not supported.

    Returns:
        None
    """
    file_extension = object_path.split(".")[-1].lower()
    if file_extension is None:
        raise ValueError(f"Unsupported file type: {object_path}")

    if file_extension == "usdz":
        # install usdz io package
        dirname = os.path.dirname(os.path.realpath(__file__))
        usdz_package = os.path.join(dirname, "io_scene_usdz.zip")
        bpy.ops.preferences.addon_install(filepath=usdz_package)
        # enable it
        addon_name = "io_scene_usdz"
        bpy.ops.preferences.addon_enable(module=addon_name)
        # import the usdz
        from io_scene_usdz.import_usdz import import_usdz

        import_usdz(context, filepath=object_path, materials=True, animations=True)
        return None

    # load from existing import functions
    import_function = IMPORT_FUNCTIONS[file_extension]

    if file_extension == "blend":
        import_function(directory=object_path, link=False)
    elif file_extension in {"glb", "gltf"}:
        import_function(filepath=object_path, merge_vertices=True)
    else:
        import_function(filepath=object_path)


def scene_bbox(
    single_obj: Optional[bpy.types.Object] = None, ignore_matrix: bool = False
) -> Tuple[Vector, Vector]:
    """Returns the bounding box of the scene.

    Taken from Shap-E rendering script
    (https://github.com/openai/shap-e/blob/main/shap_e/rendering/blender/blender_script.py#L68-L82)

    Args:
        single_obj (Optional[bpy.types.Object], optional): If not None, only computes
            the bounding box for the given object. Defaults to None.
        ignore_matrix (bool, optional): Whether to ignore the object's matrix. Defaults
            to False.

    Raises:
        RuntimeError: If there are no objects in the scene.

    Returns:
        Tuple[Vector, Vector]: The minimum and maximum coordinates of the bounding box.
    """
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in get_scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))

    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")

    return Vector(bbox_min), Vector(bbox_max)

def get_object_with_children_bounding_box(obj):
    # Initialize min/max corners of the bounding box
    min_co = Vector((float('inf'), float('inf'), float('inf')))
    max_co = Vector((float('-inf'), float('-inf'), float('-inf')))
    
    found = False
    
    # Check the object itself if it has bound_box (meshes, curves, etc.)
    if hasattr(obj, 'bound_box') and obj.type != 'EMPTY':
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)

            # Skip invalid coordinates
            if any(math.isinf(coord) or math.isnan(coord) for coord in world_corner):
                continue

            min_co = Vector((min(min_co.x, world_corner.x), 
                            min(min_co.y, world_corner.y), 
                            min(min_co.z, world_corner.z)))
            max_co = Vector((max(max_co.x, world_corner.x), 
                            max(max_co.y, world_corner.y), 
                            max(max_co.z, world_corner.z)))
        found = True
    
    # Recursively process all children
    for child in obj.children:
        child_min, child_max, child_found = get_object_with_children_bounding_box(child)
        if child_found:
            min_co = Vector((min(min_co.x, child_min.x),
                            min(min_co.y, child_min.y),
                            min(min_co.z, child_min.z)))
            max_co = Vector((max(max_co.x, child_max.x),
                            max(max_co.y, child_max.y),
                            max(max_co.z, child_max.z)))
            found = True

    return min_co, max_co, found

def get_scene_root_objects() -> Generator[bpy.types.Object, None, None]:
    """Returns all root objects in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all root objects in the
            scene.
    """
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def get_scene_meshes() -> Generator[bpy.types.Object, None, None]:
    """Returns all meshes in the scene.

    Yields:
        Generator[bpy.types.Object, None, None]: Generator of all meshes in the scene.
    """
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)) and obj.name != 'Plane':
            yield obj

def get_all_mesh_objects(root):
    """Recursively collect all mesh objects under a root (including itself if it's a mesh)."""
    objs = []
    def rec(obj):
        if obj.type == 'MESH':
            objs.append(obj)
        for child in obj.children:
            rec(child)
    rec(root)
    return objs

def mesh_objects_intersect(mesh_a, mesh_b):
    """Return True if two mesh objects intersect, using bmesh boolean ops."""
    # Duplicate the objects and bring their mesh into local (world) space
    temp_a = mesh_a.copy()
    temp_a.data = mesh_a.data.copy()
    temp_a.matrix_world = mesh_a.matrix_world
    temp_b = mesh_b.copy()
    temp_b.data = mesh_b.data.copy()
    temp_b.matrix_world = mesh_b.matrix_world
    bpy.context.collection.objects.link(temp_a)
    bpy.context.collection.objects.link(temp_b)

    # Apply transforms
    bpy.context.view_layer.update()
    for obj in [temp_a, temp_b]:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        obj.select_set(False)

    # Build bmsh
    bm_a = bmesh.new()
    bm_b = bmesh.new()
    bm_a.from_mesh(temp_a.data)
    bm_b.from_mesh(temp_b.data)

    # — build BVH trees and test for any overlapping faces —
    tree_a = BVHTree.FromBMesh(bm_a)
    tree_b = BVHTree.FromBMesh(bm_b)
    overlaps = tree_a.overlap(tree_b)
    is_intersecting = bool(overlaps)

    # Clean up
    bm_a.free()
    bm_b.free()
    bpy.data.objects.remove(temp_a)
    bpy.data.objects.remove(temp_b)
    return is_intersecting

def objects_collide(obj0, obj1, x, y, z):
    """Move root objects to (0, y, 0) and (0, -y, 0), check for collision among all mesh children."""
    loc0, loc1 = obj0.location.copy(), obj1.location.copy()
    obj0.location = (x, y, z)
    obj1.location = (-x, -y, 0)
    bpy.context.view_layer.update()

    meshes0 = get_all_mesh_objects(obj0)
    meshes1 = get_all_mesh_objects(obj1)
    found = False
    # Check all mesh pairs
    for m0 in meshes0:
        for m1 in meshes1:
            if mesh_objects_intersect(m0, m1):
                found = True
                break
        if found:
            break
    # Restore positions
    obj0.location = loc0
    obj1.location = loc1
    bpy.context.view_layer.update()
    return found

def normalize_scene(objects_info=None, scale_target=1.0, positive_z=False, multi_object_config=None) -> None:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.
    """
    
    # objects_info = []
    # scene_root_objects = list(get_scene_root_objects())
    # scene_root_objects = [obj for obj in scene_root_objects if obj.type != 'CAMERA' and obj.type != 'LIGHT'] 

    # get scale factor. all objects have the same scale factor
    scale_factor = 1e5
    for object_info in objects_info:
        obj = object_info['object']
        bbox_min, bbox_max, _ = get_object_with_children_bounding_box(obj)
        scale_factor = min(scale_factor, scale_target / max(bbox_max - bbox_min))

    # normalize each object
    for i in range(len(objects_info)):

        obj = objects_info[i]['object']
        gravity_info = objects_info[i]['gravity_info'] if objects_info[i]['gravity_info'] is not None else None # [0] is the object filepath, [1] is the gravity info

        bbox_min, bbox_max, _ = get_object_with_children_bounding_box(obj)

        obj.scale *= scale_factor

        bpy.context.view_layer.update()
        bpy.context.evaluated_depsgraph_get().update()
        bbox_min, bbox_max, _ = get_object_with_children_bounding_box(obj)

        offset = -(bbox_min + bbox_max) / 2
        if positive_z:
            offset.z = -bbox_min.z

        obj.location += offset
        bpy.context.view_layer.update()

        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        obj.rotation_mode = 'XYZ'
        bpy.context.view_layer.update()

        bbox_min, bbox_max, _ = get_object_with_children_bounding_box(obj)

        objects_info[i]['tall'] = bbox_max.z - bbox_min.z
        objects_info[i]['bottom'] = (bbox_max.x - bbox_min.x) * (bbox_max.y - bbox_min.y)

        print(f"Object {obj.name} has been normalized")

    # def sort_key(x):
    #     return x['bottom'] if multi_object_config == 'up_down' else -x['tall']
    # objects_info = sorted(objects_info, key=sort_key)

    return objects_info, scale_factor

def get_light_elevation(light_type, parallel_elevation, object_height, shadow_width, object_translation, light_distance):
    if light_type == "SUN":
        return parallel_elevation
    elif light_type == "SPOT" or light_type == "AREA" or light_type == "POINT":
        # solve rsin(theta) / h = shadow_width * canvas_radius / ((shadow_width - translate_ratio) * canvas_radius + rcos(theta))
        def equation(theta):
            lhs = object_height / light_distance * np.sin(np.radians(theta))
            rhs = shadow_width / (shadow_width - object_translation + light_distance * np.cos(np.radians(theta)))
            return lhs - rhs
        theta = fsolve(equation, np.degrees(math.asin(object_height / light_distance)))
        return theta[0] if 0 < theta[0] < 180 else theta[0] - 180
    else:
        raise ValueError(f"Invalid light type: {light_type}")

def setup_rigid_body_physics(obj, obj_type='ACTIVE', mass=1.0, collision_shape='MESH', sensitivity=0.001,
                            damping_translation=0.5, damping_rotation=0.5, 
                            use_deactivation=False,
                            deactivate_linear_velocity=10.0, deactivate_angular_velocity=10.0):
    """Setup rigid body physics for an object"""
    # Select the object
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Add rigid body physics
    bpy.ops.rigidbody.object_add()
    
    # Configure rigid body settings
    obj.rigid_body.type = obj_type
    if obj_type == 'ACTIVE':
        obj.rigid_body.mass = mass
    
    # Set collision shape to MESH for accurate collision
    obj.rigid_body.collision_shape = collision_shape
    
    # Set collision margin (sensitivity)
    obj.rigid_body.collision_margin = sensitivity

    if obj_type == 'ACTIVE':
        # Set damping
        obj.rigid_body.linear_damping = damping_translation
        obj.rigid_body.angular_damping = damping_rotation

        # Set deactivation velocity
        obj.rigid_body.use_deactivation = use_deactivation
        obj.rigid_body.deactivate_linear_velocity = deactivate_linear_velocity
        obj.rigid_body.deactivate_angular_velocity = deactivate_angular_velocity
    
    # Deselect object
    obj.select_set(False)
    
    return obj

def setup_rigid_body_world(gravity=(0, 0, -9.81)):
    """Setup the rigid body world with gravity"""
    scene = bpy.context.scene
    
    # Enable rigid body world
    bpy.ops.rigidbody.world_add()
    
    # Set gravity
    scene.gravity = gravity
    
    # Set simulation range
    scene.frame_start = 1
    scene.frame_end = 250  # Adjust as needed
    
    return scene.rigidbody_world

def run_physics_simulation(end_frame=250):
    """Run the physics simulation"""
    scene = bpy.context.scene
    
    # Set timeline
    scene.frame_start = 1
    scene.frame_end = end_frame
    scene.frame_set(1)
    
    # Run simulation by going through all frames
    for frame in range(1, end_frame + 1):
        scene.frame_set(frame)
        bpy.context.view_layer.update()
    
    print(f"Physics simulation completed for {end_frame} frames")

def get_final_transform(obj, frame=None):
    """Get the final position and rotation of the object"""
    scene = bpy.context.scene
    
    if frame is None:
        frame = scene.frame_end
    
    # Go to the final frame
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    
    # Get world transform
    final_pos = obj.matrix_world.translation.copy()
    final_rot = obj.matrix_world.to_euler()
    
    # Convert rotation to degrees
    rot_degrees = [math.degrees(angle) for angle in final_rot]
    
    return {
        'position': {
            'x': final_pos.x,
            'y': final_pos.y,
            'z': final_pos.z
        },
        'rotation_radians': {
            'x': final_rot.x,
            'y': final_rot.y,
            'z': final_rot.z
        },
        'rotation_degrees': {
            'x': rot_degrees[0],
            'y': rot_degrees[1],
            'z': rot_degrees[2]
        }
    }

def animate_object(object_filepath, initial_height=1.0, simulation_frames=250):
    """Main automation function"""
    print("Starting Blender physics automation...")
    
    # Clear the scene
    reset_scene()
    print("Scene cleared")
    
    # Create or import object
    load_object(object_filepath)
    objects_info, _ = normalize_scene(objects_info=[{'object': bpy.context.active_object, 'gravity_info': None, 'filepath': object_filepath}], 
                                      scale_target=1.0, positive_z=True, multi_object_config=None)
    target_obj = objects_info[0]['object']
    
    # Create ground plane
    bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, 0))
    ground_plane = bpy.context.active_object
    ground_plane.name = "GroundPlane"
    print("Created ground plane")
    
    # Lift the object
    target_obj.location = (0, 0, initial_height)
    print(f"Lifted object to height: {initial_height}")
    
    # Setup rigid body world
    setup_rigid_body_world()
    print("Setup rigid body world with gravity")
    
    # Add rigid body physics to target object (active)
    setup_rigid_body_physics(target_obj, 'ACTIVE', mass=1.0)
    print("Added active rigid body to target object")
    
    # Add rigid body physics to ground plane (passive)
    setup_rigid_body_physics(ground_plane, 'PASSIVE')
    print("Added passive rigid body to ground plane")
    
    # Run physics simulation
    run_physics_simulation(simulation_frames)
    
    # Get final transform
    final_transform = get_final_transform(target_obj)

    return final_transform

def render_scene(
    azimuth: float | int,
    elevation: float | int,
    object_scale: float,
    object_filepaths: List[str],
    object_translate_x: float = None,
    object_translate_y: float = None,
    translate_ratio: float = None,
    output_dir: str = None,
    apply_texture: bool = False,
    focal_length: float = 35,
    fov: float = 60,
    camera_dist: float = 2,
    light_energy: float = 75000,
    light_distance: float = 10.0,
    spot_size: float = np.radians(30),
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"] = "SPOT",
    texture_path: str = None,
    dynamic_elevation: bool = False,
    shadow_length_min: float = 0.6,
    shadow_length_max: float = 1.3,
    multi_object_config: Literal["in_out", "side_by_side", "up_down"] = "in_out",
    internal_rotation: bool = False,
    sample_distribution: bool = False,
    num_keyframes: int = 10
) -> None:
    """Saves rendered images with its camera matrix and metadata of the object.

    Args:
        object_file (str): Path to the object file.
        num_renders (int): Number of renders to save of the object.
        only_northern_hemisphere (bool): Whether to only render sides of the object that
            are in the northern hemisphere. This is useful for rendering objects that
            are photogrammetrically scanned, as the bottom of the object often has
            holes.
        output_dir (str): Path to the directory where the rendered images and metadata
            will be saved.

    Returns:
        None
    """
    reset_scene()

    object_names = []
    objects_info = []

    for object_filepath in object_filepaths:
        load_object(object_filepath)
        object_name = bpy.context.active_object.name
        object_names.append(object_name)
        objects_info.append({
            'object': bpy.context.active_object,
            'filepath': object_filepath,
            'gravity_info': None
        })

    # normalize the scene
    objects_info, scale_factor = normalize_scene(objects_info=objects_info, scale_target=object_scale, positive_z=True, multi_object_config=multi_object_config)

    # Set up cameras
    cam = scene.objects["Camera"]

    cam.data.lens = focal_length
    cam.data.sensor_width = 2 * focal_length * math.tan(math.radians(fov/2))

    # save the scale factor
    if not os.path.exists(os.path.join(output_dir, 'scale_factor.txt')):
        with open(os.path.join(output_dir, 'scale_factor.txt'), 'w') as f:
            f.write(str(scale_factor))

    # set camera
    set_camera_look_at_center(center_distance=camera_dist)

    # randomize the lighting
    # 1. remove the light object
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    world = bpy.data.worlds["World"]
    bg_node = world.node_tree.nodes["Background"]
    bg_node.inputs["Strength"].default_value = 7.5
    
    # add a sun light and set the energy
    bpy.ops.object.light_add(type=light_type, location=(0, 0, 0))
    bpy.context.object.data.energy = light_energy
    bpy.context.object.name = "Light"
    if light_type == "SPOT":
        bpy.context.object.data.spot_blend = 0
        bpy.context.object.data.spot_size = spot_size

    # add a floor with name "Plane"
    bpy.ops.object.select_all(action="DESELECT")
    canvas_radius = camera_dist*math.tan(math.radians(fov/2))
    bpy.ops.mesh.primitive_plane_add(size=2*canvas_radius, location=(0, 0, 0))
    ground_plane = bpy.context.active_object
    ground_plane.name = "GroundPlane"

    frame_start, frame_end = 10000000, -10000000
    for obj in bpy.data.objects:
        if obj.animation_data and obj.animation_data.nla_tracks:
            for track in obj.animation_data.nla_tracks:
                frame_start = min(frame_start, min([strip.frame_start for strip in track.strips]))
                frame_end = max(frame_end, max([strip.frame_end for strip in track.strips]))

    frame_start, frame_end = int(frame_start), int(frame_end)

    def render_one_scene(azimuth, elevation,
                        texture_path = None,
                        save_shadow_path = None, save_object_path = None, save_texture_path = None,
                        save_animation_path = None):

        '''
        Render a single scene with the given parameters.
        When there are two objects, enable_gravity will animate the second object; when there is one object, enable_gravity will not do anything because we already baked the gravity.
        '''

        # set the sun light rotation
        bpy.ops.object.select_all(action="DESELECT")
        obj = bpy.data.objects["Light"]
        obj.rotation_euler = (0, np.radians(90 - elevation), np.radians(azimuth))
        obj.location = (light_distance * np.cos(np.radians(azimuth)) * np.cos(np.radians(elevation)), light_distance * np.sin(np.radians(azimuth)) * np.cos(np.radians(elevation)), light_distance * np.sin(np.radians(elevation)))
        frame_to_render = list(range(frame_start, frame_end + 1, (frame_end - frame_start) // num_keyframes))

        saved_image_paths = []
        
        for frame_cnt in range(frame_start, frame_end + 1):

            scene.frame_set(frame_cnt)
            bpy.context.view_layer.update()
            if frame_cnt not in frame_to_render:
                continue

            if apply_texture:
                mat = bpy.data.materials.new(name="GroundPlane")
                mat.use_nodes = True
                nodes = mat.node_tree.nodes
                nodes.clear()

                # --- Create necessary Nodes ---
                output_node = nodes.new(type='ShaderNodeOutputMaterial')
                diffuse_node = nodes.new(type='ShaderNodeBsdfDiffuse')
                emission_node = nodes.new(type='ShaderNodeEmission')
                tex_image_node = nodes.new(type='ShaderNodeTexImage')
                light_path_node = nodes.new(type='ShaderNodeLightPath')
                mix_shader_node = nodes.new(type='ShaderNodeMixShader')
                tex_image_node.image = bpy.data.images.load(texture_path)

                # --- Link the Nodes ---
                links = mat.node_tree.links
                links.new(tex_image_node.outputs['Color'], diffuse_node.inputs['Color'])
                links.new(tex_image_node.outputs['Color'], emission_node.inputs['Color'])
                links.new(diffuse_node.outputs['BSDF'], mix_shader_node.inputs[2])  # Bottom shader = visible to light
                links.new(emission_node.outputs['Emission'], mix_shader_node.inputs[1])  # Top shader = visible to camera
                links.new(light_path_node.outputs['Is Camera Ray'], mix_shader_node.inputs['Fac'])
                links.new(mix_shader_node.outputs['Shader'], output_node.inputs['Surface'])

                # --- Assign the Material to the Plane ---
                plane = bpy.data.objects['GroundPlane']
                plane.data.materials.append(mat)


                scene.render.filepath = save_texture_path.replace('.png', f'_{frame_cnt:04d}.png')
                bpy.ops.render.render(write_still=True)
                saved_image_paths.append(scene.render.filepath)

                # remove the material
                plane.data.materials.clear()
        
            else:
                scene.render.filepath = save_shadow_path.replace('.png', f'_{frame_cnt:04d}.png')
                bpy.ops.render.render(write_still=True)
                saved_image_paths.append(scene.render.filepath)

                # hide the plane, do not render the plane
                bpy.ops.object.select_all(action="DESELECT")
                ground_plane = bpy.data.objects["GroundPlane"]
                ground_plane.select_set(True)
                ground_plane.hide_render = True
            
                scene.render.filepath = save_object_path.replace('.png', f'_{frame_cnt:04d}.png')
                bpy.ops.render.render(write_still=True)

                ground_plane.hide_render = False

        if save_animation_path is not None:
            os.makedirs(os.path.dirname(save_animation_path), exist_ok=True)
            images_to_video(saved_image_paths, save_animation_path)
            
        scene.frame_set(frame_start)
        bpy.context.view_layer.update()

        return

    if not dynamic_elevation:
        save_shadow_path = os.path.join(output_dir, f'object_shadow.png')
        save_object_path = os.path.join(output_dir, f'object.png')
        save_texture_path = texture_path.replace('line_drawing_processed', 'shadow_art') if texture_path is not None else None
        if not apply_texture and (os.path.exists(save_object_path) and os.path.exists(save_shadow_path)) or (apply_texture and os.path.exists(save_texture_path)):
            return
        assert len(objects_info) == 1, "Only support 1 object"
        obj = objects_info[0]['object']
        obj.location = (canvas_radius * object_translate_x, canvas_radius * object_translate_y, 0)
        obj.rotation_euler = (0, 0, math.atan2(object_translate_y, object_translate_x))
        render_one_scene(azimuth, elevation, texture_path, save_shadow_path, save_object_path, save_texture_path)
    else: # elevation and azimuth are ints
        scene_params_list = []
        if apply_texture:
            image_names = os.listdir(os.path.join(output_dir, 'line_drawing_processed'))
            for image_name in image_names:
                azimuth_this, elevation_this = float(image_name.split('azi')[1][:6]), float(image_name.split('ele')[1][:6])
                if internal_rotation:
                    internal_rotation_angle_this = float(image_name.split('rot')[1][:6])
                else:
                    internal_rotation_angle_this = None
                scene_params_list.append((azimuth_this, elevation_this, internal_rotation_angle_this))
        else:
            azimuths = np.linspace(0, 360, azimuth, endpoint=False).tolist()
            shadow_lengths = np.linspace(shadow_length_min, shadow_length_max, elevation) * canvas_radius
            scene_height = objects_info[0]['tall']
            elevations = [get_light_elevation(light_type, np.degrees(math.atan2(scene_height, shadow_length)), scene_height, 
                                            shadow_length, translate_ratio * canvas_radius, light_distance) for shadow_length in shadow_lengths]
            if sample_distribution:
                def score_metric(irregularity_data, coeff=50.0):
                    # return np.exp((irregularity_data['convexity'] - irregularity_data['compactness']) + ((irregularity_data['area'] / 1e4) ** 0.25) * 30.0)
                    return np.exp(coeff * (irregularity_data['fractal'] + irregularity_data['convexity'] - 2)) # -2 for numerical stability

                def get_fitter(irregularity_data_path, elevation=None):
                    with open(irregularity_data_path, 'r') as f:
                        irregularity_data = json.load(f)
                    
                    x_data, y_data = [], []
                    for image_name, irregularity_data in irregularity_data.items():
                        if elevation is not None and abs(float(image_name.split('ele')[-1][:6]) - elevation) > 0.01:
                            continue
                        x_this, y_this = float(image_name.split('rot')[-1][:6]), score_metric(irregularity_data)
                        x_data.append(x_this)
                        y_data.append(y_this)
                        if x_this == 0:
                            x_data.append(360)
                            y_data.append(y_this)
                    x_data = np.array(x_data)
                    y_data = np.array(y_data)
                    y_data = y_data / np.sum(y_data)
                    fitter = InverseTransformSampler(x_data, y_data)

                    return fitter

                for elevation in elevations:
                    fitter = get_fitter(os.path.join(output_dir, 'irregularity.json'), elevation)
                    samples = fitter.sample(n_samples=len(azimuths))
                    fitter.plot_distribution(samples, save_path=os.path.join(output_dir, f'distribution_ele{elevation:06.2f}.png'))
                    scene_params_list.extend([(azimuth, elevation, sample) for (azimuth, sample) in zip(azimuths, samples)])
            else:
                scene_params_list = [(azimuth, elevation, None if not internal_rotation else np.random.randint(0, 360)) for azimuth in azimuths for elevation in elevations]

        processed_azi_eles = set()
        os.makedirs(os.path.join(output_dir, 'shadow_art'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'object_renderings'), exist_ok=True)
        processed_image_names = os.listdir(os.path.join(output_dir, 'shadow_art')) if apply_texture else os.listdir(os.path.join(output_dir, 'object_renderings'))
        for image_name in processed_image_names:
            azi, ele = float(image_name.split('azi')[1][:6]), float(image_name.split('ele')[1][:6])
            processed_azi_eles.add((azi, ele))

        def is_processed(azi, ele, processed_set, tol=0.01):
            return any(math.isclose(azi, p_azi, abs_tol=tol) and math.isclose(ele, p_ele, abs_tol=tol)
                    for (p_azi, p_ele) in processed_set)

        for (azimuth, elevation, internal_rotation_angle) in scene_params_list:

            if is_processed(azimuth, elevation, processed_azi_eles):
                continue

            def format_savename(azimuth, elevation, internal_rotation_angle=None):
                if internal_rotation_angle is None:
                    return f'azi{azimuth:06.2f}_ele{elevation:06.2f}'
                return f'azi{azimuth:06.2f}_ele{elevation:06.2f}_rot{float(internal_rotation_angle):06.2f}'
                
            save_shadow_art_path = os.path.join(output_dir, 'shadow_art', f'{format_savename(azimuth, elevation, internal_rotation_angle)}.png')
            texture_path = os.path.join(output_dir, 'line_drawing_processed', f'{format_savename(azimuth, elevation, internal_rotation_angle)}.png')

            light_elevation = elevation
            light_azimuth = azimuth
            object_rotation = internal_rotation_angle + azimuth if internal_rotation_angle is not None else azimuth
            object_azimuth = azimuth
            save_shadow_path = os.path.join(output_dir, 'object_shadow_renderings', f'{format_savename(azimuth, elevation, internal_rotation_angle)}.png')
            save_object_path = os.path.join(output_dir, 'object_renderings', f'{format_savename(azimuth, elevation, internal_rotation_angle)}.png')
            save_animation_path = os.path.join(output_dir, 'animation', f'{format_savename(azimuth, elevation, internal_rotation_angle)}.mp4') if apply_texture else None

            obj = objects_info[0]['object']
            obj.location = (canvas_radius * translate_ratio * np.cos(np.radians(object_azimuth)), canvas_radius * translate_ratio * np.sin(np.radians(object_azimuth)), 0)
            obj.rotation_euler = (0, 0, np.radians(object_rotation))
            render_one_scene(light_azimuth, light_elevation, texture_path, save_shadow_path, save_object_path, save_shadow_art_path, save_animation_path)
    return

def render_shadow_sampling(
    elevation_count: int,
    render_count: int,
    object_scale: float,
    object_filepaths: List[str],
    translate_ratio: float = None,
    output_dir: str = None,
    focal_length: float = 35,
    fov: float = 60,
    camera_dist: float = 2,
    light_energy: float = 75000,
    light_distance: float = 10.0,
    spot_size: float = np.radians(30),
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"] = "SPOT",
    shadow_length_min: float = 0.6,
    shadow_length_max: float = 1.3,
    multi_object_config: Literal["in_out", "side_by_side", "up_down"] = "in_out",
) -> None:
    """Saves rendered images with its camera matrix and metadata of the object.

    Args:
        object_file (str): Path to the object file.
        num_renders (int): Number of renders to save of the object.
        only_northern_hemisphere (bool): Whether to only render sides of the object that
            are in the northern hemisphere. This is useful for rendering objects that
            are photogrammetrically scanned, as the bottom of the object often has
            holes.
        output_dir (str): Path to the directory where the rendered images and metadata
            will be saved.

    Returns:
        None
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    reset_scene()

    object_names = []
    objects_info = []

    for object_filepath in object_filepaths:
        load_object(object_filepath)
        object_name = bpy.context.active_object.name
        object_names.append(object_name)
        objects_info.append({
            'object': bpy.context.active_object,
            'filepath': object_filepath,
            'gravity_info': None,
        })

    # normalize the scene
    objects_info, _ = normalize_scene(objects_info=objects_info, scale_target=object_scale, positive_z=True, multi_object_config=multi_object_config)

    # Set up cameras
    cam = scene.objects["Camera"]

    cam.data.lens = focal_length
    cam.data.sensor_width = 2 * focal_length * math.tan(math.radians(fov/2))

    # set camera
    set_camera_look_at_center(center_distance=camera_dist * 1.5) # ensure everything is visible, for rendering shadow only

    # randomize the lighting
    # 1. remove the light object
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="LIGHT")
    bpy.ops.object.delete()

    world = bpy.data.worlds["World"]
    bg_node = world.node_tree.nodes["Background"]
    bg_node.inputs["Strength"].default_value = 7.5
    
    # add a sun light and set the energy
    bpy.ops.object.light_add(type=light_type, location=(0, 0, 0))
    bpy.context.object.data.energy = light_energy
    bpy.context.object.name = "Light"
    if light_type == "SPOT":
        bpy.context.object.data.spot_blend = 0
        bpy.context.object.data.spot_size = spot_size

    # add a floor with name "Plane"
    canvas_radius = camera_dist*math.tan(math.radians(fov/2))
    bpy.ops.mesh.primitive_plane_add(size=4*canvas_radius, location=(0, 0, 0))
    bpy.context.object.name = "GroundPlane"

    # shadow_lengths = [(shadow_length_min + shadow_length_max) / 2 * canvas_radius] 
    shadow_lengths = np.linspace(shadow_length_min, shadow_length_max, elevation_count) * canvas_radius
    scene_height = objects_info[0]['tall']
    elevations = [get_light_elevation(light_type, np.degrees(math.atan2(scene_height, shadow_length)), scene_height, 
                                    shadow_length, translate_ratio * canvas_radius, light_distance) for shadow_length in shadow_lengths]

    azimuth = 0

    os.makedirs(os.path.join(output_dir, 'object_shadow_sampling'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'object_sampling'), exist_ok=True)

    for elevation in elevations:

        # set the sun light rotation
        bpy.ops.object.select_all(action="DESELECT")
        obj = bpy.data.objects["Light"]
        obj.rotation_euler = (0, np.radians(90 - elevation), np.radians(azimuth))
        obj.location = (light_distance * np.cos(np.radians(azimuth)) * np.cos(np.radians(elevation)), light_distance * np.sin(np.radians(azimuth)) * np.cos(np.radians(elevation)), light_distance * np.sin(np.radians(elevation)))

        for object_rotation in np.linspace(0, 360, render_count, endpoint=False).tolist():
            object_shadow_image_path = os.path.join(output_dir, 'object_shadow_sampling', f'azi{azimuth:06.2f}_ele{elevation:06.2f}_rot{object_rotation:06.2f}.png')
            object_image_path = os.path.join(output_dir, 'object_sampling', f'azi{azimuth:06.2f}_ele{elevation:06.2f}_rot{object_rotation:06.2f}.png')
            rendered_object_image_path = os.path.join(output_dir, 'object_sampling', f'azi{azimuth:06.2f}_ele{elevations[0]:06.2f}_rot{object_rotation:06.2f}.png')

            if os.path.exists(object_shadow_image_path) and os.path.exists(object_image_path):
                print(f"Skipping {object_shadow_image_path} and {object_image_path} because they already exist")
                continue
            obj = objects_info[0]['object']
            obj.location = (canvas_radius * translate_ratio * np.cos(np.radians(azimuth)), canvas_radius * translate_ratio * np.sin(np.radians(azimuth)), 0)
            obj.rotation_euler = (0, 0, np.radians(object_rotation))

            scene.render.filepath = object_shadow_image_path
            bpy.ops.render.render(write_still=True)

            # hide the plane, do not render the plane
            if not os.path.exists(rendered_object_image_path):
                bpy.ops.object.select_all(action="DESELECT")
                ground_plane = bpy.data.objects["GroundPlane"]
                ground_plane.select_set(True)
                ground_plane.hide_render = True
            
                scene.render.filepath = object_image_path
                bpy.ops.render.render(write_still=True)
                ground_plane.hide_render = False
            else: # copy, because they are the same
                shutil.copy(rendered_object_image_path, object_image_path)

    return

# ------------------------------------------------------------

scene = bpy.context.scene
render = scene.render

# Set render settings
render.engine = "CYCLES"
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 1024
render.resolution_y = 1024
render.resolution_percentage = 100

# Set cycles settings
scene.cycles.device = "GPU"
scene.cycles.samples = 256
scene.cycles.diffuse_bounces = 2
scene.cycles.glossy_bounces = 2
scene.cycles.transparent_max_bounces = 2
scene.cycles.transmission_bounces = 2
scene.cycles.filter_width = 1.0
scene.cycles.use_denoising = False
scene.render.film_transparent = True

bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"  # or "OPENCL"
bpy.context.preferences.addons["cycles"].preferences.get_devices()

def run_render(json_data):

    params = json.loads(json_data)

    obj_path = params["object_filepath"]
    obj_scale = params["object_scale"] if 'object_scale' in params else 0.8
    obj_tx = params["object_translate_x"] if 'object_translate_x' in params else None
    obj_ty = params["object_translate_y"] if 'object_translate_y' in params else None
    azimuth = params["azimuth"] if 'azimuth' in params else None
    elevation = params["elevation"] if 'elevation' in params else None
    render_count = params["render_count"] if 'render_count' in params else 60
    translate_ratio = params["translate_ratio"] if 'translate_ratio' in params else 0.8
    dynamic_elevation = params["dynamic_elevation"] if 'dynamic_elevation' in params else False
    output_root = params["output_root"] if 'output_root' in params else None
    apply_texture = params["apply_texture"] if 'apply_texture' in params else False
    texture_path = params["texture_path"] if 'texture_path' in params else None
    resolution = params["resolution"] if 'resolution' in params else 1024
    shadow_length_min = params["shadow_length_min"] if 'shadow_length_min' in params else 0.6
    shadow_length_max = params["shadow_length_max"] if 'shadow_length_max' in params else 1.3
    multi_object_config = params["multi_object_config"] if 'multi_object_config' in params else None
    internal_rotation = params["internal_rotation"] if 'internal_rotation' in params else False
    sample_distribution = params["sample_distribution"] if 'sample_distribution' in params else False
    shadow_sampling = params["shadow_sampling"] if 'shadow_sampling' in params else False
    camera_dist = params['camera_dist'] if 'camera_dist' in params else 2
    do_animate = params["do_animate"] if 'do_animate' in params else False
    initial_height = params["initial_height"] if 'initial_height' in params else 1.0
    simulation_frames = params["simulation_frames"] if 'simulation_frames' in params else 250
    num_keyframes = params["num_keyframes"] if 'num_keyframes' in params else 10
    focal_length = 35
    fov = 60

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution

    if shadow_sampling:
        render_shadow_sampling(
            elevation_count=elevation,
            render_count=render_count,
            object_scale=obj_scale,
            object_filepaths=obj_path,
            output_dir=output_root,
            focal_length=focal_length,
            fov=fov,
            camera_dist=camera_dist,
            translate_ratio=translate_ratio,
            shadow_length_min=shadow_length_min,
            shadow_length_max=shadow_length_max,
            multi_object_config=multi_object_config,
        )
        return

    if do_animate:
        result = animate_object(object_filepath=obj_path, initial_height=initial_height, simulation_frames=simulation_frames)
        save_path = os.path.join(os.path.dirname(obj_path), 'animate.json')
        with open(save_path, 'w') as f:
            json.dump(result, f)
        return

    if dynamic_elevation:
        assert obj_tx is None and obj_ty is None, "object_translate_x and object_translate_y must be None for dynamic elevation"
        assert type(azimuth) == int and type(elevation) == int, "azimuth and elevation must be integers (the number of azimuth and elevation steps)"
        assert translate_ratio is not None, "translate_ratio must not be None for dynamic elevation"
        if apply_texture:
            assert texture_path is None and output_root is not None, "read texture from the output directory, texture_path should be None"

    print(f"Rendering object {obj_path} at azimuth {azimuth}, elevation {elevation}, scale {obj_scale}, "
        f"translate {obj_tx}, {obj_ty}, dynamic_elevation {dynamic_elevation}")

    render_scene(
        azimuth=azimuth,
        elevation=elevation,
        internal_rotation=internal_rotation,
        dynamic_elevation=dynamic_elevation,
        object_scale=obj_scale,
        object_translate_x=obj_tx,
        object_translate_y=obj_ty,
        apply_texture=apply_texture,
        object_filepaths=obj_path,
        output_dir=output_root,
        focal_length=focal_length,
        fov=fov,
        camera_dist=camera_dist,
        texture_path=texture_path,
        translate_ratio=translate_ratio,
        shadow_length_min=shadow_length_min,
        shadow_length_max=shadow_length_max,
        multi_object_config=multi_object_config,
        sample_distribution=sample_distribution,
        num_keyframes=num_keyframes
    )

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost')
    parser.add_argument('--port', type=int, default=65432)
    parser.add_argument('--run_server', action='store_true')
    parser.add_argument('--json_data', type=str, default=None)

    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    assert args.json_data is not None or args.run_server, "json_data must be provided when run_server is False"

    if args.run_server:

        # Set up socket server
        sock = socket.socket()
        sock.bind((args.host, args.port))
        sock.listen(1)

        print(f"Blender render server started at {args.host}:{args.port}")

        while True:
            conn, addr = sock.accept()
            data = conn.recv(4096).decode()
            if not data:
                continue
            if data.strip() == "quit":
                break

            if data.strip() == "test":
                conn.sendall(b'Connection OK')
                continue

            try:
                run_render(data)
                conn.sendall(b'done')

            except Exception as e:
                tb = traceback.format_exc()
                conn.sendall(f'error: {tb}'.encode())

            # Deselect everything
            bpy.ops.object.select_all(action='DESELECT')

            # Select only objects to delete (e.g., all meshes, excluding Camera and Light)
            for obj in bpy.data.objects:
                if obj.type in ['MESH', 'EMPTY'] and obj.name not in ['Camera', 'Light']:
                    obj.select_set(True)

            bpy.ops.object.delete(use_global=False)

            # Clear out all unused data to prevent memory leak
            for block in bpy.data.meshes:
                if block.users == 0:
                    bpy.data.meshes.remove(block)
            for block in bpy.data.images:
                if block.users == 0:
                    bpy.data.images.remove(block)
            for block in bpy.data.materials:
                if block.users == 0:
                    bpy.data.materials.remove(block)
            for block in bpy.data.textures:
                if block.users == 0:
                    bpy.data.textures.remove(block)
            for block in bpy.data.lights:
                if block.users == 0:
                    bpy.data.lights.remove(block)

            for img in bpy.data.images:
                if "Render Result" in img.name:
                    img.user_clear()
                    bpy.data.images.remove(img)

        sock.close()

    else:
        run_render(args.json_data)

if __name__ == "__main__":
    main()