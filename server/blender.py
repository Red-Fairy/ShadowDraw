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
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
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

def get_angle(number_string, num_digits=6):
    if number_string[0] == '-':
        try:
            number_string = -float(number_string[1:num_digits+1])
        except:
            number_string = -float(number_string[1:num_digits])
    else:
        number_string = float(number_string[:num_digits])
    return number_string

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

    if file_extension == "ply":

        obj = bpy.context.active_object
        obj.rotation_mode = 'XYZ'
        obj.rotation_euler = (math.pi / 2, 0, 0)
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # add vertex colors
        me = obj.data

        # --- 1) Find the vertex color layer name ---
        vcol_name = (me.color_attributes.active_color.name 
                    if getattr(me.color_attributes, "active_color", None) 
                    else me.color_attributes[0].name)

        if vcol_name is None:
            raise RuntimeError("No vertex color/color attribute layer found on the active object.")

        # --- 2) Create/assign a material with nodes ---
        mat = bpy.data.materials.new(name="VColor_Mat")
        mat.use_nodes = True

        # Clear existing nodes for a clean setup
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        nodes.clear()

        # Nodes: Color Attribute (or Attribute) -> Principled BSDF -> Material Output
        out_node = nodes.new("ShaderNodeOutputMaterial")
        out_node.location = (400, 0)

        bsdf = nodes.new("ShaderNodeBsdfPrincipled")
        bsdf.location = (150, 0)

        attr_node = nodes.new("ShaderNodeAttribute")
        attr_node.attribute_name = vcol_name

        attr_node.label = "Color Attribute"
        attr_node.location = (-150, 0)

        links.new(attr_node.outputs.get("Color"), bsdf.inputs.get("Base Color"))
        links.new(bsdf.outputs.get("BSDF"), out_node.inputs.get("Surface"))

        # Optional: set some Principled defaults similar to your screenshot
        bsdf.inputs["Roughness"].default_value = 0.5
        bsdf.inputs["Alpha"].default_value = 1.0

        # --- 3) Assign the material to the object ---
        if obj.data.materials:
            obj.data.materials[0] = mat
        else:
            obj.data.materials.append(mat)

        print(f"Material '{mat.name}' created. Using vertex color layer: '{vcol_name}'.")

    return


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

def normalize_scene(objects_info=None, scale_target=1.0, positive_z=False, try_put_down=True) -> None:
    """Normalizes the scene by scaling and translating it to fit in a unit cube centered
    at the origin.
    """
    
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

        # bbox_min, bbox_max, _ = get_object_with_children_bounding_box(obj)

        obj.scale *= scale_factor
        bpy.context.view_layer.update()
        bpy.context.evaluated_depsgraph_get().update()
        bbox_min, bbox_max, _ = get_object_with_children_bounding_box(obj)

        offset = -(bbox_min + bbox_max) / 2
        if positive_z:
            offset.z = -bbox_min.z
            
        obj.location += offset
        bpy.context.view_layer.update()

        # select the empty object
        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        # bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='BOUNDS')

        bbox_min, bbox_max, _ = get_object_with_children_bounding_box(obj)
        dist_x, dist_y, dist_z = bbox_max - bbox_min

        # rotate the object if z is loarge
        update_bbox = False
        obj.rotation_mode = 'XYZ'
        if try_put_down:
            if dist_z > 2 * dist_y or dist_z > 2 * dist_x:
                if dist_y > dist_x:
                    obj.rotation_euler = (0, np.pi / 2, 0)
                else:
                    obj.rotation_euler = (np.pi / 2, 0, 0)
                update_bbox = True
            bpy.context.view_layer.update()

        if gravity_info is not None:
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            # bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='BOUNDS')
            obj.rotation_euler = (gravity_info['rotation_radians']['x'], gravity_info['rotation_radians']['y'], gravity_info['rotation_radians']['z'])
            update_bbox = True
        bpy.context.view_layer.update()

        if update_bbox:

            if gravity_info is not None:
                bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
                # bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='BOUNDS')

            bbox_min, bbox_max, _ = get_object_with_children_bounding_box(obj)
            offset = -(bbox_min + bbox_max) / 2
            if positive_z:
                offset.z = -bbox_min.z
            obj.location += offset
        
            bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
            # bpy.ops.object.origin_set(type='ORIGIN_CURSOR', center='BOUNDS')

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

def animate_object(object_filepath, initial_height=1.0, simulation_frames=250, try_put_down=True):
    """Main automation function"""
    print("Starting Blender physics automation...")
    
    # Clear the scene
    reset_scene()
    print("Scene cleared")
    
    # Create or import object
    load_object(object_filepath)
    objects_info, _ = normalize_scene(objects_info=[{'object': bpy.context.active_object, 'gravity_info': None}], scale_target=1.0, positive_z=True, try_put_down=try_put_down)
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

def combine_objects(
    object_scale: float,
    object_filepaths: List[str],
    camera_dist: float = 2,
    fov: float = 60,
    output_dir: str = None,
    try_put_down: bool = True,
    simulation_frames: int = 250,
    ):

    reset_scene()

    object_names = []
    objects_info = []

    for i, object_filepath in enumerate(object_filepaths):
        load_object(object_filepath)
        object_name = bpy.context.active_object.name
        object_names.append(object_name)
        objects_info.append({
            'object': bpy.context.active_object,
            'filepath': object_filepath,
            'gravity_info': None,
        })

    # normalize the scene
    objects_info, scale_factor = normalize_scene(objects_info=objects_info, scale_target=object_scale, positive_z=True, try_put_down=try_put_down)

    # add a floor with name "Plane"
    bpy.ops.object.select_all(action="DESELECT")
    canvas_radius = camera_dist*math.tan(math.radians(fov/2))
    bpy.ops.mesh.primitive_plane_add(size=2*canvas_radius, location=(0, 0, 0))
    ground_plane = bpy.context.active_object
    ground_plane.name = "GroundPlane"

    setup_rigid_body_world()
    setup_rigid_body_physics(ground_plane, 'PASSIVE', collision_shape='CONVEX_HULL', sensitivity=0.001)
    setup_rigid_body_physics(objects_info[0]['object'], 'PASSIVE', collision_shape='MESH', sensitivity=0.001)
    for object_info in objects_info[1:]:
        setup_rigid_body_physics(object_info['object'], 'ACTIVE', mass=1.0, collision_shape='CONVEX_HULL', sensitivity=0.001, use_deactivation=True)

    # add a empty parent object to the objects
    bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
    parent_empty = bpy.context.active_object
    parent_empty.name = "ParentEmpty"
    for object_info in objects_info:
        object_info['object'].parent = parent_empty
    bpy.context.view_layer.update()

    for i, object_info in enumerate(objects_info):
        object_info['object'].location = (0, 0, sum([object_info['tall'] for object_info in objects_info[:i]]) + 0.1 * i)

    run_physics_simulation(end_frame=simulation_frames)
    save_dict = {}
    for i, obj in enumerate(objects_info[1:]):
        obj_relative_transform = obj['object'].matrix_local.copy() # compute relative transform with the parent empty
        save_dict[f'obj{i+1}'] = {'position': {'x': obj_relative_transform.translation.x, 'y': obj_relative_transform.translation.y, 'z': obj_relative_transform.translation.z}, 
                                'rotation': {'x': obj_relative_transform.to_euler().x, 'y': obj_relative_transform.to_euler().y, 'z': obj_relative_transform.to_euler().z}}
    with open(os.path.join(output_dir, 'gravity.json'), 'w') as f:
        json.dump(save_dict, f)

    # export the objects as a single obj file
    bpy.ops.object.select_all(action="DESELECT")
    for object_info in objects_info:
        object_info['object'].select_set(True)
    bpy.ops.wm.obj_export(filepath=os.path.join(output_dir, 'combined_objects.obj'), 
                            export_selected_objects=True,
                            forward_axis="NEGATIVE_Z",
                            up_axis="Y")
    return

def compute_light_elevations(
    output_dir: str,
    elevation_count: int,
    object_scale: float,
    object_filepaths: List[str],
    translate_ratio: float = None,
    fov: float = 60,
    camera_dist: float = 2,
    light_distance: float = 10.0,
    light_type: Literal["POINT", "SUN", "SPOT", "AREA"] = "SPOT",
    shadow_length_min: float = 0.6,
    shadow_length_max: float = 1.3,
    use_gravity: bool = False,
    try_put_down: bool = True
) -> None:

    reset_scene()

    objects_info = []

    for i, object_filepath in enumerate(object_filepaths):
        load_object(object_filepath)
        objects_info.append({
            'object': bpy.context.active_object,
            'filepath': object_filepath,
            'gravity_info': json.load(open(os.path.join(output_dir, f'animate_{i}.json'))) if use_gravity else None,
        })

    # normalize the scene
    objects_info, scale_factor = normalize_scene(objects_info=objects_info, scale_target=object_scale, positive_z=True, try_put_down=try_put_down)

    canvas_radius = camera_dist*math.tan(math.radians(fov/2))
    shadow_lengths = np.linspace(shadow_length_min, shadow_length_max, elevation_count) * canvas_radius
    scene_height = objects_info[0]['tall']
    elevations = [get_light_elevation(light_type, np.degrees(math.atan2(scene_height, shadow_length)), scene_height, 
                                    shadow_length, translate_ratio * canvas_radius, light_distance) for shadow_length in shadow_lengths]

    save_path = os.path.join(output_dir, 'light_elevations.json')
    with open(save_path, 'w') as f:
        json.dump(elevations, f)

    return

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
    read_scene_params: bool = False,
    multi_object_config: Literal["in_out", "side_by_side", "up_down"] = "in_out",
    internal_rotation: bool = False,
    sample_distribution: bool = False,
    use_gravity: bool = False,
    try_put_down: bool = True,
    add_material: bool = False
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

    if dynamic_elevation:
        num_renders = elevation * azimuth
    else:
        num_renders = 1

    for i, object_filepath in enumerate(object_filepaths):
        load_object(object_filepath)
        object_name = bpy.context.active_object.name
        object_names.append(object_name)
        objects_info.append({
            'object': bpy.context.active_object,
            'filepath': object_filepath,
            'gravity_info': json.load(open(os.path.join(output_dir, f'animate_{i}.json'))) if use_gravity else None,
        })

    # normalize the scene
    objects_info, scale_factor = normalize_scene(objects_info=objects_info, scale_target=object_scale, positive_z=True, try_put_down=try_put_down)

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

    if use_gravity and len(objects_info) >= 2:
        setup_rigid_body_world()
        setup_rigid_body_physics(ground_plane, 'PASSIVE', collision_shape='CONVEX_HULL', sensitivity=0.001)
        setup_rigid_body_physics(objects_info[0]['object'], 'PASSIVE', collision_shape='MESH', sensitivity=0.001)
        for object_info in objects_info[1:]:
            setup_rigid_body_physics(object_info['object'], 'ACTIVE', mass=1.0, collision_shape='CONVEX_HULL', sensitivity=0.001, use_deactivation=True)

    def render_one_scene(azimuth, elevation,
                        texture_path = None,
                        save_shadow_path = None, save_object_path = None, save_texture_path = None):

        '''
        Render a single scene with the given parameters.
        When there are two objects, enable_gravity will animate the second object; when there is one object, enable_gravity will not do anything because we already baked the gravity.
        '''

        # set the sun light rotation
        bpy.ops.object.select_all(action="DESELECT")
        obj = bpy.data.objects["Light"]
        obj.rotation_euler = (0, np.radians(90 - elevation), np.radians(azimuth))
        obj.location = (light_distance * np.cos(np.radians(azimuth)) * np.cos(np.radians(elevation)), light_distance * np.sin(np.radians(azimuth)) * np.cos(np.radians(elevation)), light_distance * np.sin(np.radians(elevation)))

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

            scene.render.filepath = save_texture_path
            bpy.ops.render.render(write_still=True)

            # remove the material
            plane.data.materials.clear()

        else:
            scene.render.filepath = save_shadow_path
            bpy.ops.render.render(write_still=True)
            
            # hide the plane, do not render the plane
            bpy.ops.object.select_all(action="DESELECT")
            ground_plane = bpy.data.objects["GroundPlane"]
            ground_plane.select_set(True)
            ground_plane.hide_render = True
        
            scene.render.filepath = save_object_path
            bpy.ops.render.render(write_still=True)
            ground_plane.hide_render = False
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

        if len(objects_info) >= 2:
            # add a empty parent object to the objects
            bpy.ops.object.empty_add(type='PLAIN_AXES', align='WORLD', location=(0, 0, 0), scale=(1, 1, 1))
            parent_empty = bpy.context.active_object
            parent_empty.name = "ParentEmpty"
            for object_info in objects_info:
                object_info['object'].parent = parent_empty
            bpy.context.view_layer.update()

            for i, object_info in enumerate(objects_info):
                x_this = np.random.uniform(-0.1, 0.1) if i != 0 else 0
                y_this = np.random.uniform(-0.1, 0.1) if i != 0 else 0
                object_info['object'].location = (x_this, y_this, sum([object_info['tall'] for object_info in objects_info[:i]]) + 0.1 * i)

        if add_material:  # create a plain color material for the object
            mat = bpy.data.materials.new(name="ObjectMaterial")
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            nodes.clear()
            
            output_node = nodes.new(type='ShaderNodeOutputMaterial')
            diffuse_node = nodes.new(type='ShaderNodeBsdfDiffuse')
            links = mat.node_tree.links
            links.new(diffuse_node.outputs['BSDF'], output_node.inputs['Surface'])
            
            diffuse_node.inputs['Color'].default_value = (0, 0, 139/255, 1.0)  # light blue with alpha
            
            for object_info in objects_info:
                object_info['object'].data.materials.clear()
                object_info['object'].data.materials.append(mat)

        scene_params_list = []
        if apply_texture:
            image_names = [x for x in os.listdir(os.path.join(output_dir, 'line_drawing_processed')) if x.endswith('.png')]
            for image_name in image_names:
                azimuth_this, elevation_this = get_angle(image_name.split('azi')[1]), get_angle(image_name.split('ele')[1])
                if internal_rotation:
                    if len(objects_info) == 1:
                        internal_rotation_angle_this = get_angle(image_name.split('rot')[1])
                    else:
                        internal_rotation_angle_this = [get_angle(image_name.split('rot')[i]) for i in range(1, len(objects_info) + 1)]
                else:
                    internal_rotation_angle_this = None
                scene_params_list.append((azimuth_this, elevation_this, internal_rotation_angle_this))
        elif read_scene_params:
            json_file = json.load(open(os.path.join(output_dir, 'object_params_optimized.json')))
            azimuths = np.linspace(0, 360, azimuth, endpoint=False).tolist() * elevation
            assert len(azimuths) == len(json_file), "Number of azimuths and object parameters must match"
            scene_params_list = [(azimuth, json_file[i]['light_elevation'], json_file[i]['object_self_rotation']) for i, azimuth in enumerate(azimuths)]
        else:
            azimuths = np.linspace(0, 360, azimuth, endpoint=False).tolist()
            shadow_lengths = np.linspace(shadow_length_min, shadow_length_max, elevation) * canvas_radius
            if len(objects_info) == 1 or multi_object_config != "up_down":
                scene_height = objects_info[0]['tall']
            else:
                scene_height = sum([object_info['tall'] for object_info in objects_info])

            elevations = [get_light_elevation(light_type, np.degrees(math.atan2(scene_height, shadow_length)), scene_height, 
                                            shadow_length, translate_ratio * canvas_radius, light_distance) for shadow_length in shadow_lengths]
            if sample_distribution:
                def score_metric(irregularity_data, coeff=25.0):
                    # return np.exp((irregularity_data['convexity'] - irregularity_data['compactness']) + ((irregularity_data['area'] / 1e4) ** 0.25) * 30.0)
                    return np.exp(coeff * (irregularity_data['fractal'] + irregularity_data['convexity'] - 2)) # -2 for numerical stability

                def get_fitter(irregularity_data_path, elevation=None):
                    with open(irregularity_data_path, 'r') as f:
                        irregularity_data = json.load(f)
                    
                    x_data, y_data = [], []
                    for image_name, irregularity_data in irregularity_data.items():
                        if elevation is not None and abs(get_angle(image_name.split('ele')[-1]) - elevation) > 0.01:
                            continue
                        x_this, y_this = get_angle(image_name.split('rot')[-1]), score_metric(irregularity_data)
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

                if len(objects_info) == 1:
                    for elevation in elevations:
                        fitter = get_fitter(os.path.join(output_dir, 'irregularity_0.json'), elevation)
                        samples = fitter.sample(n_samples=len(azimuths))
                        fitter.plot_distribution(samples, save_path=os.path.join(output_dir, f'plot_distribution/distribution_ele{elevation:06.2f}.png'))
                        scene_params_list.extend([(azimuth, elevation, sample) for (azimuth, sample) in zip(azimuths, samples)])
                else:
                    fitters = [get_fitter(os.path.join(output_dir, f'irregularity_{i}.json')) for i, object_info in enumerate(objects_info)]
                    for elevation in elevations:
                        samples = np.zeros((len(azimuths), len(objects_info)))
                        for i, fitter in enumerate(fitters):
                            samples[:, i] = fitter.sample(n_samples=len(azimuths))
                            fitter.plot_distribution(samples[:, i], save_path=os.path.join(output_dir, f'distribution_ele{elevation:06.2f}_obj{i}.png'))
                        scene_params_list.extend([(azimuth, elevation, sample) for (azimuth, sample) in zip(azimuths, samples)])
            else:
                scene_params_list = [(azimuth, elevation, None if not internal_rotation else np.random.randint(0, 360)) for azimuth in azimuths for elevation in elevations]

        processed_azi_eles = set()
        os.makedirs(os.path.join(output_dir, 'shadow_art'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'object'), exist_ok=True)
        if use_gravity and len(objects_info) >= 2:
            os.makedirs(os.path.join(output_dir, 'gravity'), exist_ok=True) 
        processed_image_names = os.listdir(os.path.join(output_dir, 'shadow_art')) if apply_texture else os.listdir(os.path.join(output_dir, 'object'))
        for image_name in processed_image_names:
            azi, ele = get_angle(image_name.split('azi')[1]), get_angle(image_name.split('ele')[1])
            processed_azi_eles.add((azi, ele))

        def is_processed(azi, ele, processed_set, tol=0.01):
            return any(math.isclose(azi, p_azi, abs_tol=tol) and math.isclose(ele, p_ele, abs_tol=tol)
                    for (p_azi, p_ele) in processed_set)

        for (azimuth, elevation, internal_rotation_angle) in scene_params_list:

            if len(processed_azi_eles) >= num_renders or is_processed(azimuth, elevation, processed_azi_eles):
                continue

            def format_savename(azimuth, elevation, internal_rotation_angle=None):
                if internal_rotation_angle is None:
                    return f'azi{azimuth:06.2f}_ele{elevation:06.2f}'
                elif len(objects_info) == 1:
                    return f'azi{azimuth:06.2f}_ele{elevation:06.2f}_rot{float(internal_rotation_angle):06.2f}'
                else:
                    rots = '_'.join([f'rot{float(rot):06.2f}' for rot in internal_rotation_angle])
                    return f'azi{azimuth:06.2f}_ele{elevation:06.2f}_{rots}'
                
            save_shadow_art_path = os.path.join(output_dir, 'shadow_art', f'{format_savename(azimuth, elevation, internal_rotation_angle)}.png')
            texture_path = os.path.join(output_dir, 'line_drawing_processed', f'{format_savename(azimuth, elevation, internal_rotation_angle)}.png')

            light_elevation = elevation
            light_azimuth = azimuth
            object_rotation = internal_rotation_angle + azimuth if (internal_rotation_angle is not None and len(objects_info) == 1) else azimuth
            object_azimuth = azimuth
            save_shadow_path = os.path.join(output_dir, 'object_shadow', f'{format_savename(azimuth, elevation, internal_rotation_angle)}.png')
            save_object_path = os.path.join(output_dir, 'object', f'{format_savename(azimuth, elevation, internal_rotation_angle)}.png')

            if len(objects_info) == 1:
                obj = objects_info[0]['object']
                obj.location = (canvas_radius * translate_ratio * np.cos(np.radians(object_azimuth)), canvas_radius * translate_ratio * np.sin(np.radians(object_azimuth)), 0)
                obj.rotation_euler = (0, 0, np.radians(object_rotation))
                render_one_scene(light_azimuth, light_elevation, texture_path, save_shadow_path, save_object_path, save_shadow_art_path)
            else:
                parent_empty.location = (canvas_radius * translate_ratio * np.cos(np.radians(object_azimuth)), canvas_radius * translate_ratio * np.sin(np.radians(object_azimuth)), 0)
                parent_empty.rotation_euler = (0, 0, np.radians(object_rotation))
                if internal_rotation_angle is not None:
                    for i, obj in enumerate(objects_info):
                        obj['object'].rotation_euler = (0, 0, np.radians(internal_rotation_angle[i]))
                if use_gravity:
                    initial_pos = [obj['object'].location.copy() for obj in objects_info[1:]]
                    initial_rot = [obj['object'].rotation_euler.copy() for obj in objects_info[1:]]
                    if not apply_texture: # run simulation and save the pose
                        run_physics_simulation(end_frame=75)
                        scene_height_new = scene_bbox()[1].z
                        light_elevation = math.degrees(math.atan(math.tan(math.radians(light_elevation)) * (scene_height_new / scene_height)))
                        save_dict = {}
                        for i, obj in enumerate(objects_info[1:]):
                            obj_relative_transform = obj['object'].matrix_local.copy() # compute relative transform with the parent empty
                            save_dict[f'obj{i+1}'] = {'position': {'x': obj_relative_transform.translation.x, 'y': obj_relative_transform.translation.y, 'z': obj_relative_transform.translation.z}, 
                                                    'rotation': {'x': obj_relative_transform.to_euler().x, 'y': obj_relative_transform.to_euler().y, 'z': obj_relative_transform.to_euler().z}}
                        save_object_path = os.path.join(output_dir, 'object', f'{format_savename(azimuth, light_elevation, internal_rotation_angle)}.png')
                        save_shadow_path = os.path.join(output_dir, 'object_shadow', f'{format_savename(azimuth, light_elevation, internal_rotation_angle)}.png')
                        save_gravity_path = os.path.join(output_dir, 'gravity', f'{format_savename(azimuth, light_elevation, internal_rotation_angle)}.json')
                        with open(save_gravity_path, 'w') as f:
                            json.dump(save_dict, f)
                    else:
                        save_gravity_path = os.path.join(output_dir, 'gravity', f'{format_savename(azimuth, light_elevation, internal_rotation_angle)}.json')
                        assert os.path.exists(save_gravity_path), f"Gravity info not found for {save_gravity_path}"
                        with open(save_gravity_path, 'r') as f:
                            data = json.load(f)
                        for i, obj in enumerate(objects_info[1:]):
                            obj['object'].location = (data[f'obj{i+1}']['position']['x'], data[f'obj{i+1}']['position']['y'], data[f'obj{i+1}']['position']['z'])
                            obj['object'].rotation_euler = (data[f'obj{i+1}']['rotation']['x'], data[f'obj{i+1}']['rotation']['y'], data[f'obj{i+1}']['rotation']['z'])
                   
                    render_one_scene(light_azimuth, light_elevation, texture_path, save_shadow_path, save_object_path, save_shadow_art_path)
                    scene.frame_set(1)
                    bpy.context.view_layer.update()
                    for i, obj in enumerate(objects_info[1:]):
                        obj['object'].location = initial_pos[i]
                        obj['object'].rotation_euler = initial_rot[i]
                else:
                    render_one_scene(light_azimuth, light_elevation, texture_path, save_shadow_path, save_object_path, save_shadow_art_path)
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
    read_scene_params = params["read_scene_params"] if 'read_scene_params' in params else False
    internal_rotation = params["internal_rotation"] if 'internal_rotation' in params else False
    sample_distribution = params["sample_distribution"] if 'sample_distribution' in params else False
    use_gravity = params["use_gravity"] if 'use_gravity' in params else False
    camera_dist = params['camera_dist'] if 'camera_dist' in params else 2
    do_animate = params["do_animate"] if 'do_animate' in params else False
    animate_save_path = params["animate_save_path"] if 'animate_save_path' in params else None
    initial_height = params["initial_height"] if 'initial_height' in params else 1.0
    simulation_frames = params["simulation_frames"] if 'simulation_frames' in params else 250
    compute_light_elevation = params["compute_light_elevation"] if 'compute_light_elevation' in params else False
    focal_length = 35
    fov = 60
    try_put_down = params["try_put_down"] if 'try_put_down' in params else True
    add_material = params["add_material"] if 'add_material' in params else False
    object_index = params["object_index"] if 'object_index' in params else 0 # used for multi-object rendering during shadow sampling

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution

    if compute_light_elevation:
        compute_light_elevations(   
            output_dir=output_root,
            elevation_count=elevation,
            object_scale=obj_scale,
            object_filepaths=obj_path,
            translate_ratio=translate_ratio,
            fov=fov,
            camera_dist=camera_dist,
            shadow_length_min=shadow_length_min,
            shadow_length_max=shadow_length_max,
            use_gravity=use_gravity,
            try_put_down=try_put_down
        )
        return

    if do_animate:
        if type(obj_path) == list:
            combine_objects(
                object_scale=obj_scale,
                object_filepaths=obj_path,
                camera_dist=camera_dist,
                fov=fov,
                output_dir=output_root,
                try_put_down=try_put_down,
            )
        else:
            result = animate_object(object_filepath=obj_path, initial_height=initial_height, simulation_frames=simulation_frames, try_put_down=try_put_down)
            with open(animate_save_path, 'w') as f:
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
        read_scene_params=read_scene_params,
        sample_distribution=sample_distribution,
        use_gravity=use_gravity,
        try_put_down=try_put_down,
        add_material=add_material
    )

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--json_data', type=str, required=True)

    argv = sys.argv[sys.argv.index("--") + 1 :]
    args = parser.parse_args(argv)

    run_render(args.json_data)

if __name__ == "__main__":
    main()