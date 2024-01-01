import blenderproc as bproc
import argparse
import math
import os
import numpy as np
import time
import urllib.request
from matplotlib import pyplot as plt
import blenderproc.python.renderer.RendererUtility as RendererUtility
from blenderproc.scripts.saveAsImg import save_array_as_image
from blenderproc.scripts.visHdf5Files import vis_data

# blenderproc run --custom-blender-path={BLD_DIR} --blender-install-path={BLD_DIR} render_single_glb.py --object_path {glb_path} \
#  --output_dir {out_folder} --engine CYCLES --num_images 12 --camera_dist 1.2

import bpy
from mathutils import Vector

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="./views")
parser.add_argument(
    "--engine", type=str, default="BLENDER_EEVEE", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--num_images", type=int, default=12)
parser.add_argument("--camera_dist", type=float, default=1.5)

#argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args()

bproc.init()

context = bpy.context
scene = context.scene
render = scene.render

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 32
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True

def add_lighting() -> None:
    # delete the default light
    #bpy.data.objects["Light"].select_set(True)
    #bpy.ops.object.delete()
    # add a new light
    bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = 30000
    bpy.data.objects["Area"].location[2] = 0.5
    bpy.data.objects["Area"].scale[0] = 100
    bpy.data.objects["Area"].scale[1] = 100
    bpy.data.objects["Area"].scale[2] = 100


def reset_scene() -> None:
    """Resets the scene to a clean state."""
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


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")

def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
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

def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale * 0.8
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")

def setup_camera():
    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.lens = 35
    cam.data.sensor_width = 32
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    return cam, cam_constraint

def sample_camera_loc(phi=None, theta=None, r=3.5):
    #phi = np.random.uniform(np.pi / 3, np.pi / 3 * 2)
    #theta = np.random.uniform(0, np.pi * 2)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.array([x, y, z])

def save_images(object_file: str) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)
    #reset_scene()
    # load the object
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    normalize_scene()
    add_lighting()
    cam, cam_constraint = setup_camera()
    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    views = [[np.pi / 3, np.pi / 6],
            [np.pi / 3, np.pi / 6 * 4],
            [np.pi / 3, np.pi / 6 * 7],
            [np.pi / 3, np.pi / 6 * 10],
            [np.pi / 2, np.pi / 6 * 2],
            [np.pi / 2, np.pi / 6 * 5],
            [np.pi / 2, np.pi / 6 * 8],
            [np.pi / 2, np.pi / 6 * 11],
            [np.pi / 3 * 2, 0],
            [np.pi / 3 * 2, np.pi / 2],
            [np.pi / 3 * 2, np.pi],
            [np.pi / 3 * 2, np.pi / 2 * 3]]

    for i in range(args.num_images):
        # Sample random camera location around the object
        location = sample_camera_loc(views[i][0], views[i][1], args.camera_dist)
        
        # Compute rotation based on vector going from location towards the location of the ShapeNet object
        rotation_matrix = bproc.camera.rotation_from_forward_vec([0,0,0] - location)
        # Add homog cam pose based on location an rotation
        cam2world_matrix = bproc.math.build_transformation_mat(location, rotation_matrix)
        bproc.camera.add_camera_pose(cam2world_matrix)
    
    RendererUtility.set_cpu_threads(10)
    bproc.renderer.enable_normals_output()
    bproc.renderer.enable_depth_output(activate_antialiasing=False)
    bproc.renderer.set_output_format(enable_transparency=True)

    data = bproc.renderer.render(verbose=True)
    for index, image in enumerate(data["colors"]):
        render_path = os.path.join(args.output_dir, f"{index:03d}.png")
        save_array_as_image(image, "colors", render_path)
    for index, image in enumerate(data["normals"]):
        render_path = os.path.join(args.output_dir, f"{index:03d}_normal.png")
        save_array_as_image(image, "normals", render_path)
    #cam_states = []
    #for frame in range(bproc.utility.num_frames()):
    #    cam_states.append({
    #        "cam2world": bproc.camera.get_camera_pose(frame),
    #        "cam_K": bproc.camera.get_intrinsics_as_K_matrix()
    #    })
    # Adds states to the data dict
    #print(cam_states)

    for index, image in enumerate(data["depth"]):
        render_path = os.path.join(args.output_dir, f"{index:03d}_depth.png")
        #save_array_as_image(image, "depth", render_path)
        #print(image.min(), image[image < 100].max())
        #vis_data("depth", image, None, "", save_to_file=render_path, depth_max = image[image < 100].max())
        plt.imsave(render_path, image, cmap='gray', vmax=image[image < 100].max())


def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


if __name__ == "__main__":
    try:
        start_i = time.time()
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path
        save_images(local_path)
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        # delete the object if it was downloaded
        if args.object_path.startswith("http"):
            os.remove(local_path)
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)
