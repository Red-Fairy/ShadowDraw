import bpy
import os
import argparse
import sys
from typing import Dict, Callable

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--object_path', type=str, required=True)

    argv = sys.argv[sys.argv.index("--") + 1:]
    args = parser.parse_args(argv)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    IMPORT_FUNCTIONS[args.object_path.split('.')[-1]](filepath=args.object_path)

    # Deselect all first
    bpy.ops.object.select_all(action='DESELECT')

    # Get all mesh objects
    mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']

    if not mesh_objs:
        print("No mesh objects found in the scene.")
        return

    # Select all mesh objects
    for obj in mesh_objs:
        obj.select_set(True)

    # Make the first one active
    bpy.context.view_layer.objects.active = mesh_objs[0]

    # Join them into one
    bpy.ops.object.join()

    out_dir = os.path.dirname(args.object_path)
    base_name = os.path.splitext(os.path.basename(args.object_path))[0]
    obj_path = os.path.join(out_dir, base_name + ".obj")
    obj_info_path = os.path.join(out_dir, base_name + '.txt')

    # save the length of mesh_objs
    with open(obj_info_path, 'w') as f:
        f.write(str(len(mesh_objs)))

    # Export the single joined object
    bpy.ops.wm.obj_export(
        filepath=obj_path,
        export_materials=True,
        export_selected_objects=False,  # Export everything
        forward_axis='NEGATIVE_Z',
        up_axis='Y'
    )

    print(f"Exported: {obj_path} and {base_name}.mtl")

if __name__ == "__main__":
    main()