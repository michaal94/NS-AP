'''
Based on:
clevr-dataset-gen
https://github.com/facebookresearch/clevr-dataset-gen
'''

from __future__ import print_function
import os
import sys
# As its called from blender we need to add the following
# to import from inside project
PROJECT_PATH = os.path.abspath('..')
sys.path.insert(0, PROJECT_PATH)
print("Python paths:")
print(sys.path)
import random
import argparse
import json
import tempfile
import pycocotools.mask as mask_utils
from collections import Counter
import cv2
import numpy as np
from mathutils.bvhtree import BVHTree

"""
Renders random scenes using Blender, each with with a random number of objects;
each object has a random size, position, color, and shape. Objects will be
nonintersecting but may partially occlude each other. Output images will be
written to disk as PNGs, and we will also write a JSON file for each image with
ground-truth scene information, such as given properties and encoded segmentation mask.

This file expects to be run from Blender like this:

blender --background --python render_images.py -- [arguments to this script]
"""


INSIDE_BLENDER = True
try:
    import bpy
    from mathutils import Vector
except ImportError:
    INSIDE_BLENDER = False
if INSIDE_BLENDER:
    try:
        import scene_generation.utils as utils
        from scene_generation.config import Config
    except ImportError:
        print("\nERROR")
        print("Running render_images.py from Blender and cannot import utils.py.")
        print("You may need to add a .pth file to the site-packages of Blender's")
        print("bundled python with a command like this:\n")
        print("echo $PWD >> $BLENDER/$VERSION/python/lib/python3.7/site-packages/shop_vrb.pth")
        print("\nWhere $BLENDER is the directory where Blender is installed, and")
        print("$VERSION is your Blender version (such as 2.81).")
        sys.exit(1)

parser = argparse.ArgumentParser()

def main():
    output_dir = '../output_ycb_images'
    with open('data/object_properties_ycb.json', 'r') as f:
        properties = json.load(f)

    # Load the main blendfile
    bpy.ops.wm.open_mainfile(filepath='data/base_scene_panda_w_gripper.blend')

    # Set render arguments so we can get pixel coordinates later.
    # We use functionality specific to the CYCLES renderer so BLENDER_RENDER
    # cannot be used.
    render_args = bpy.context.scene.render
    render_args.engine = "CYCLES"
    
    render_args.resolution_x = 1280
    render_args.resolution_y = 720
    render_args.resolution_percentage = 100
    render_args.tile_x = 2048
    render_args.tile_y = 2048

    # Blender changed the API for enabling CUDA at some point
    if bpy.app.version < (2, 78, 0):
        bpy.context.user_preferences.system.compute_device_type = 'CUDA'
        bpy.context.user_preferences.system.compute_device = 'CUDA_0'
    elif bpy.app.version < (2, 80, 0):
        cycles_prefs = bpy.context.user_preferences.addons['cycles'].preferences
        cycles_prefs.compute_device_type = 'CUDA'
    else:
        cycles_prefs = bpy.context.preferences.addons['cycles'].preferences
        cycles_prefs.compute_device_type = 'CUDA'
        bpy.context.preferences.addons["cycles"].preferences.get_devices()

    bpy.data.worlds['World'].cycles.sample_as_light = True

    # Some CYCLES-specific stuff
    bpy.context.scene.cycles.blur_glossy = 2.0
    bpy.context.scene.cycles.samples = 64
    bpy.context.scene.cycles.transparent_min_bounces = 8
    bpy.context.scene.cycles.transparent_max_bounces = 8
    bpy.context.scene.cycles.device = 'GPU'

    active_camera = 'cam1'

    bpy.context.scene.camera = bpy.data.objects[active_camera]

    robot_init_config = {
        "link0": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        "link1": [[0.0, 0.0, 0.333], [0.9998691665585041, 0.0, 0.0, -0.01617559166158651]],
        "link2": [[0.0, 0.0, 0.333], [0.647006369915434, -0.655895620968082, -0.26419866463495667, -0.2852766329176867]],
        "link3": [[-0.22616912899960642, 0.007319712068981651, 0.5535673296360697], [0.9211564702100625, -0.0034309816290204334, -0.3885734024190059, -0.021672486883529282]],
        "link4": [[-0.16865986164921432, 0.004245669122323428, 0.6126391878156058], [0.49856024692570233, 0.5247347667369178, 0.47949380143858245, -0.4961620694338823]],
        "link5": [[0.21060701209724964, -0.012983945938004638, 0.7132474819242763], [0.7234700497508134, 0.013331628061234172, 0.6900123257021316, -0.017214679373375934]],
        "link6": [[0.21060701209724964, -0.012983945938004638, 0.7132474819242763], [0.7086797143528601, 0.7047506275063948, -0.03306426614070607, 0.0025238460071294044]],
        "link7": [[0.2984134797702896, -0.01677031557442508, 0.7176845575107164], [-0.007315500647490094, 0.9108730620682938, -0.41190932142096953, 0.024237557119552347]]
    }

    # robot_init_config = {
    #         'link0': ([0., 0., 0.], [1., 0., 0., 0.]), 
    #         'link1': ([0.   , 0.   , 0.333], [ 0.99996686,  0.        ,  0.        , -0.00814159]), 
    #         'link2': ([0.   , 0.   , 0.333], [ 0.70709987, -0.70705704,  0.00838687, -0.003127  ]),
    #         'link3': ([ 2.35034211e-03, -3.82748638e-05,  6.48991257e-01], [ 9.99982226e-01,  4.32315698e-05,  3.71916331e-03, -4.65983623e-03]),
    #         'link4': ([ 0.08484448, -0.00080711,  0.64837757], [ 0.27288862,  0.27897062,  0.64977019, -0.65230813]),
    #         'link5': ([ 0.41813879, -0.00390273,  0.44060951], [3.90227048e-01, 9.98689328e-03, 9.20664290e-01, 6.15478348e-04]),
    #         'link6': ([ 0.41813879, -0.00390273,  0.44060951], [ 0.70392049,  0.71001676, -0.01843918,  0.00566933]),
    #         'link7': ([ 0.50607329, -0.00550457,  0.4436024 ], [-0.01058263,  0.91810927, -0.39593709,  0.01404281])
    #     }
   
    if 'Robot' in bpy.data.collections:
        for obj in bpy.data.collections['Robot'].all_objects:
            if obj.parent is None:
                pos, quat = robot_init_config[obj.name]
                obj.location = pos
                obj.rotation_mode = 'QUATERNION'
                obj.rotation_quaternion = quat

    gripper_init_config = {
        "right_gripper": [[0.30375779051602575, -0.017477514602558836, 0.6113210853401088], [0.019597010807198743, 0.7289621446269386, 0.6840856788268902, 0.01602912637262325]],
        "leftfinger": [[0.30434751879722177, -0.04986274786754622, 0.5590987820440172], [0.002522875278178263, 0.9991756981050939, -0.031732453282900426, 0.02519148318733432]],
        "rightfinger": [[0.30842735675402855, 0.01421639481154386, 0.558877721325398], [0.002522875278178263, 0.9991756981050939, -0.031732453282900426, 0.02519148318733432]]
    }

    if 'Gripper' in bpy.data.collections:
        for obj in bpy.data.collections['Gripper'].all_objects:
            if obj.parent is None:
                pos, quat = gripper_init_config[obj.name]
                obj.location = pos
                obj.rotation_mode = 'QUATERNION'
                obj.rotation_quaternion = quat

    x_min = 0.17
    x_mid = 0.44
    x_max = 0.67
    y_min = -0.24
    y_mid = 0.0
    y_max = 0.24

    positions = {
        'tl': [x_min, y_min],
        'tm': [x_min, y_mid],
        'tr': [x_min, y_max],
        'ml': [x_mid, y_min],
        'mm': [x_mid, y_mid],
        'mr': [x_mid, y_max],
        'bl': [x_max, y_min],
        'bm': [x_max, y_mid],
        'br': [x_max, y_max]
    }

    for fname in [
        'bleach_cleanser',
        # 'bowl',
        # 'cracker_box',
        # 'foam_brick',
        # 'mug',
        # 'mustard_bottle',
        # 'potted_meat_can',
        # 'sugar_box',
        # 'tomato_soup_can',
    ]:
        for pos in ['tl', 'tm', 'tr', 'ml', 'mm', 'mr', 'bl', 'bm', 'br']:
            for rot in range(0, 360, 15):
                output_image = os.path.join(
                    output_dir, f'{fname}_{pos}_{rot:03d}.png'
                )
                print(output_image)
                render_args.filepath = output_image
                utils.add_object('data/shapes_ycb', fname, 1.0, positions[pos], theta=rot)
                obj = bpy.context.object
                x, y, z = obj.dimensions
                bbox = {
                    'x': x,
                    'y': y,
                    'z': z
                }
                obj.rotation_mode = 'QUATERNION'
                while True:
                    try:
                        bpy.ops.render.render(write_still=True)
                        break
                    except Exception as e:
                        print(e)
                with open(os.path.join(
                    output_dir, f'{fname}_{pos}_{rot:03d}.json'
                ), 'w') as f:
                    json.dump(
                        {
                            'file': fname,
                            'name': properties[fname]['name'],
                            'shape': properties[fname]['shape'],
                            'material': properties[fname]['material1'],
                            'colour': properties[fname]['color1'],
                            'file_path': f'{fname}_{pos}_{rot:03d}.png',
                            'pos': list(obj.location),
                            'ori': list(obj.rotation_quaternion),
                            'bbox': bbox
                        },
                        f, indent=4
                    )
                utils.delete_object(obj)


def add_random_objects(scene_struct, num_objects, config, camera):
    """
    Add random objects to the current blender scene
    """

    # Load the property file
    with open(config.properties_json, 'r') as f:
        properties = json.load(f)
        color_name_to_rgba = {}
        for name, rgb in properties['colors'].items():
            rgba = [float(c) / 255.0 for c in rgb] + [1.0]
            color_name_to_rgba[name] = rgba
        object_mapping = [(v, k) for k, v in properties['shapes'].items()]
        size_mapping = properties['sizes']

    white = [0.88, 0.88, 0.88, 1.0]

    with open(config.object_props, 'r') as f:
        object_properties = json.load(f)

    positions = []
    weights = []
    objects = []
    blender_objects = []
    bvh_list = []

    used = []

    if 'Robot' in bpy.data.collections:
        correct = True
        if 'link0' in bpy.data.objects:
            obj = bpy.data.objects['link0']
        elif 'base_link' in bpy.data.objects:
            obj = bpy.data.objects['base_link']
        else:
            correct = False
        if correct:
            target_mat = obj.matrix_world
            target_vert = [target_mat @ v.co for v in obj.data.vertices]
            target_poly = [p.vertices for p in obj.data.polygons]
            target_bvh = BVHTree.FromPolygons(target_vert, target_poly)
            bvh_list.append(target_bvh)
    for i in range(num_objects):
        # Choose random color and shape
        obj_name, obj_name_out = random.choice(object_mapping)
        for _ in range(100):
            if obj_name in used:
                obj_name, obj_name_out = random.choice(object_mapping)
            else:
                break
        used.append(obj_name)
        if 'soda_can' in obj_name_out:
            obj_name_out = 'soda_can'
        color_name, rgba = random.choice(list(color_name_to_rgba.items()))

        # Choose a random size
        if object_properties[obj_name]['change_size']:
            size_name = random.choice(list(size_mapping.keys()))
            r = size_mapping[size_name]
            if size_name == "bigger":
                size_name = object_properties[obj_name]['size1']
            else:
                size_name = object_properties[obj_name]['size2']
        else:
            size_name = object_properties[obj_name]['size1']
            r = size_mapping["bigger"]

        num_tries = 0
        while True:
            # If we try and fail to place an object too many times, then delete all
            # the objects in the scene and start over.
            num_tries += 1
            # print(num_tries)
            if num_tries > config.max_retries:
                for obj in blender_objects:
                    utils.delete_object(obj)
                del blender_objects
                del objects
                return None, None
            # x = random.uniform(-0.25, 0.25)
            x = random.uniform(0.3, 0.7)
            y = random.uniform(-0.32, 0.32)


            # x = x + 0.4
            # y = y - 0.1

            # Choose random orientation for the object.
            theta = 180.0 * random.random()

            # Check to make sure the new object is further than min_dist from all
            # other objects, and further than margin along the four cardinal directions
            margins_good = True

            pos_temp = positions.copy()
            pos_temp.append((x, y, r, theta))

            for (xx, yy, rr, th) in positions:
                dx, dy = x - xx, y - yy
                for direction_name in ['left', 'right', 'front', 'behind']:
                    direction_vec = scene_struct['directions'][direction_name]
                    assert direction_vec[2] == 0
                    margin = dx * direction_vec[0] + dy * direction_vec[1]
                    if 0 < margin < config.margin:
                        print(margin, config.margin, direction_name)
                        print('Broken margin!')
                        margins_good = False
                        break
                if not margins_good:
                    break

            # Actually add the object to the scene
            utils.add_object(config.shape_dir, obj_name, r, (x, y), theta=theta)
            obj = bpy.context.object

            dists_good, new_bvh = utils.check_intersections_bvh(bvh_list, obj, config.intersection_eps)
            dists_good = not dists_good

            if dists_good and margins_good:
                bvh_list.append(new_bvh)
                break
            else:
                utils.delete_object(obj)

        blender_objects.append(obj)
        positions.append((x, y, r, theta))

        # Attach a random material
        if object_properties[obj_name]['change_material']:
            change = random.choice([True, False])
            if change:
                mat_name_out = object_properties[obj_name]['material2']
                mat_name = mat_name_out.capitalize()
                if object_properties[obj_name]['change_color2']:
                    utils.add_material(mat_name, Color=rgba)
                else:
                    color_name = object_properties[obj_name]['color2']
                    utils.add_material(mat_name, Color=white)
            else:
                mat_name_out = object_properties[obj_name]['material1']
                if object_properties[obj_name]['change_color1']:
                    utils.add_color(rgba)
                else:
                    color_name = object_properties[obj_name]['color1']
        else:
            mat_name_out = object_properties[obj_name]['material1']
            if object_properties[obj_name]['change_color1']:
                print(obj_name)
                utils.add_color(rgba)
            else:
                color_name = object_properties[obj_name]['color1']

        # Record data about the object in the scene data structure
        pixel_coords = utils.get_camera_coords(camera, obj.location)

        obj_name_out = object_properties[obj_name]['name']

        weight = object_properties[obj_name]['weight_gt_mean'] * r
        weight_out = weight
        # for _ in range(100):
        #     rand_percentage = np.random.uniform(-config.weight_range, config.weight_range)
        #     weight_out = (1 + rand_percentage) * weight
        #     weight_out = np.around(weight_out, decimals=1)
        #     if weight_out not in weights:
        #         weights.append(weight_out)
        #         break
        stackable = object_properties[obj_name]['stackable']
        stack_base = object_properties[obj_name]['stack_base']
        pickupable = object_properties[obj_name]['pickupable']

        movability = object_properties[obj_name]['movability']
        shape = object_properties[obj_name]['shape']

        x, y, z = obj.dimensions
        bbox = {
            'x': x,
            'y': y,
            'z': z
        }
        obj.rotation_mode = 'QUATERNION'

        objects.append({
            'file': obj_name,
            'name': obj_name_out,
            'shape': shape,
            'size': size_name,
            'material': mat_name_out,
            '3d_coords': tuple(obj.location),
            'orientation': tuple(obj.rotation_quaternion),
            'rotation': theta,
            'pixel_coords': pixel_coords,
            'colour': color_name,
            'weight_gt': weight_out,
            'weight': None,
            'movability': movability,
            'bbox': bbox,
            'scale_factor': r,
            'stackable': stackable,
            'stack_base': stack_base,
            'pickupable': pickupable
        })

    # Check that all objects are at least partially visible
    # in the rendered image
    all_visible, masks, extra_masks = check_visibility(blender_objects, config.min_pixels_per_object)
    if not all_visible:
        # If any of the objects are fully occluded then start over; delete all
        # objects from the scene and place them all again.
        print('Some objects are occluded; replacing objects')
        for obj in blender_objects:
            utils.delete_object(obj)
        return add_random_objects(scene_struct, num_objects, config, camera)

    for obj, mask in zip(objects, masks):
        obj['mask'] = mask

    if 'table' in extra_masks:
        scene_struct['table'] = {}
        scene_struct['table']['mask'] = extra_masks['table']

    if 'robot' in extra_masks:
        scene_struct['robot'] = {}
        scene_struct['robot']['mask'] = extra_masks['robot']

    return objects, blender_objects


def compute_all_relationships(scene_struct, eps=0.0):
    """
    Computes relationships between all pairs of objects in the scene.

    Returns a dictionary mapping string relationship names to lists of lists of
    integers, where output[rel][i] gives a list of object indices that have the
    relationship rel with object i. For example if j is in output['left'][i] then
    object j is left of object i.
    """
    all_relationships = {}
    for name, direction_vec in scene_struct['directions'].items():
        if name == 'above' or name == 'below':
            continue
        all_relationships[name] = []
        for i, obj1 in enumerate(scene_struct['objects']):
            coords1 = obj1['3d_coords']
            related = set()
            for j, obj2 in enumerate(scene_struct['objects']):
                if obj1 == obj2:
                    continue
                coords2 = obj2['3d_coords']
                diff = [coords2[k] - coords1[k] for k in [0, 1, 2]]
                dot = sum(diff[k] * direction_vec[k] for k in [0, 1, 2])
                if dot > eps:
                    related.add(j)
            all_relationships[name].append(sorted(list(related)))
    return all_relationships


def check_visibility(blender_objects, min_pixels_per_object):
    """
    Check whether all objects in the scene have some minimum number of visible
    pixels; to accomplish this we assign random (but distinct) colors to all
    objects, and render using no lighting or shading or antialiasing; this
    ensures that each object is just a solid uniform color. We can then count
    the number of pixels of each color in the output image to check the visibility
    of each object.

    Returns True if all objects are visible and False otherwise.
    If True returns segmentation mask as well
    """
    f, path = tempfile.mkstemp(suffix='.png')
    # path = '../output/test_mask.png'
    # Render shadeless and return list of colours
    object_colors = render_shadeless(blender_objects, path)
    img = bpy.data.images.load(path)
    p = list(img.pixels)
    # Count whether the number of colours is correct - full occlusion
    color_count = Counter((p[i], p[i + 1], p[i + 2], p[i + 3]) for i in range(0, len(p), 4))
    additional_objects = 1
    # print(color_count)
    # exit()
    if 'Desk' in bpy.data.objects:
        additional_objects += 1
    if 'Robot' in bpy.data.collections:
        additional_objects += 1
    if len(color_count) != len(blender_objects) + additional_objects:
        print("Full occlusion detected")
        return False, None, None
    # Check partial occlusion
    for col, count in color_count.most_common():
        # print(col, count)
        if count < min_pixels_per_object:
            print("Partial occlusion detected")
            return False, None, None

    # Assign masks
    masks = assign_masks(object_colors, path)
    extra_masks = dict()
    if 'Desk' in bpy.data.objects:
        desk_mask = assign_masks([(80, 80, 80)], path)
        if desk_mask is not None:
            extra_masks['table'] = desk_mask[0]
    if 'Robot' in bpy.data.collections:
        robot_mask = assign_masks([(127, 127, 127)], path)
        if robot_mask is not None:
            extra_masks['robot'] = robot_mask[0]
    return True, masks, extra_masks


def render_shadeless(blender_objects, path='flat.png'):
    """
    Render a version of the scene with shading disabled and unique materials
    assigned to all objects, and return a set of all colors that should be in the
    rendered image. The image itself is written to path. This is used to ensure
    that all objects will be visible in the final rendered scene.
    """
    render_args = bpy.context.scene.render

    # Cache the render args we are about to clobber
    old_filepath = render_args.filepath
    old_filter_size = render_args.filter_size

    # Override some render settings to have flat shading
    render_args.filepath = path

    # Switch denoising state
    old_denoising_state = bpy.context.scene.node_tree.nodes["Switch"].check
    bpy.context.scene.node_tree.nodes["Switch"].check = False
    old_cycles_denoising = bpy.context.view_layer.cycles.use_denoising
    bpy.context.view_layer.cycles.use_denoising = False

    # Don't render lights
    utils.set_render(bpy.data.objects['Lamp_Key'], False)
    utils.set_render(bpy.data.objects['Lamp_Fill'], False)
    utils.set_render(bpy.data.objects['Lamp_Back'], False)
    utils.set_render(bpy.data.objects['Ground'], False)

    # Change shading and AA
    old_shading = bpy.context.scene.display.shading.light
    bpy.context.scene.display.shading.light = 'FLAT'
    old_aa = bpy.context.scene.display.render_aa
    bpy.context.scene.display.render_aa = 'OFF'

    # Cycles settings
    old_blur = bpy.context.scene.cycles.blur_glossy
    bpy.context.scene.cycles.blur_glossy = 0.0
    old_samples = bpy.context.scene.cycles.samples
    bpy.context.scene.cycles.samples = 1
    old_light_bounces = bpy.context.scene.cycles.max_bounces
    bpy.context.scene.cycles.max_bounces = 0

    # Add random shadeless materials to all objects
    object_colors = []
    new_obj = []

    def create_shadeless_copy(obj, num, colour):
        obj.select_set(state=True)
        bpy.ops.object.duplicate(linked=False, mode='INIT')
        utils.set_render(obj, False)
        mat = bpy.data.materials['Shadeless'].copy()
        mat.name = 'Shadeless_temp_%d' % num
        group_node = mat.node_tree.nodes['Group']
        r, g, b = colour
        for inp in group_node.inputs:
            if inp.name == 'Color':
                inp.default_value = (float(r) / 255, float(g) / 255, float(b) / 255, 1.0)
        for i in range(len(bpy.context.selected_objects[0].data.materials)):
            bpy.context.selected_objects[0].data.materials[i] = mat

    for obj in bpy.data.objects:
        obj.select_set(state=False)
    for i, obj in enumerate(blender_objects):
        while True:
            r, g, b = [random.randint(0, 255) for _ in range(3)]
            colour_correct = True
            colour_correct = colour_correct and ((r, g, b) not in object_colors)
            colour_correct = colour_correct and ((r, g, b) != (13, 13, 13))
            colour_correct = colour_correct and ((r, g, b) != (80, 80, 80))
            colour_correct = colour_correct and ((r, g, b) != (127, 127, 127))
            if colour_correct:
                break
        object_colors.append((r, g, b))
        create_shadeless_copy(obj, i, (r, g, b))
        new_obj.append(bpy.context.selected_objects[0])
        for o in bpy.data.objects:
            o.select_set(state=False)

    if 'Desk' in bpy.data.objects:
        i = i + 1
        create_shadeless_copy(bpy.data.objects['Desk'], i, (80, 80, 80))
        new_obj.append(bpy.context.selected_objects[0])
        for o in bpy.data.objects:
            o.select_set(state=False)

    if 'Robot' in bpy.data.collections:
        robot_parts = []
        for obj in bpy.data.collections['Robot'].all_objects:
            robot_parts.append(obj)

        for r_part in robot_parts:
            i = i + 1
            create_shadeless_copy(r_part, i, (127, 127, 127))
            new_obj.append(bpy.context.selected_objects[0])
            for o in bpy.data.objects:
                o.select_set(state=False)

    if 'Gripper' in bpy.data.collections:
        gripper_parts = []
        for obj in bpy.data.collections['Gripper'].all_objects:
            gripper_parts.append(obj)

        for g_part in gripper_parts:
            i = i + 1
            create_shadeless_copy(g_part, i, (127, 127, 127))
            new_obj.append(bpy.context.selected_objects[0])
            for o in bpy.data.objects:
                o.select_set(state=False)

    # Render the scene
    # Save gamma
    gamma = bpy.context.scene.view_settings.view_transform
    bpy.context.scene.view_settings.view_transform = 'Raw'
    bpy.ops.render.render(write_still=True)
    bpy.context.scene.view_settings.view_transform = gamma

    # Undo the above; first restore the materials to objects
    for obj in new_obj:
        obj.select_set(state=True)
        bpy.ops.object.delete()

    for obj in blender_objects:
        utils.set_render(obj, True)

    if 'Desk' in bpy.data.objects:
        utils.set_render(bpy.data.objects['Desk'], True)

    if 'Robot' in bpy.data.collections:
        for r_part in robot_parts:
            utils.set_render(r_part, True)

    if 'Gripper' in bpy.data.collections:
        for g_part in gripper_parts:
            utils.set_render(g_part, True)

    # Render lights again
    utils.set_render(bpy.data.objects['Lamp_Key'], True)
    utils.set_render(bpy.data.objects['Lamp_Fill'], True)
    utils.set_render(bpy.data.objects['Lamp_Back'], True)
    utils.set_render(bpy.data.objects['Ground'], True)

    # Set the render settings back to what they were
    render_args.filepath = old_filepath
    render_args.filter_size = old_filter_size
    bpy.context.scene.display.shading.light = old_shading
    bpy.context.scene.display.render_aa = old_aa
    bpy.context.scene.node_tree.nodes["Switch"].check = old_denoising_state
    bpy.context.view_layer.cycles.use_denoising = old_cycles_denoising
    bpy.context.scene.cycles.blur_glossy = old_blur
    bpy.context.scene.cycles.samples = old_samples
    bpy.context.scene.cycles.max_bounces = old_light_bounces

    return object_colors


def assign_masks(colors, image_path):
    masks = []
    img = cv2.imread(image_path)
    img_rgb = img[:, :, [2, 1, 0]]
    # Read shadeless render and encode masks in COCO format
    for color in colors:
        mask = np.all(img_rgb == color, axis=-1)
        mask = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        mask['counts'] = str(mask['counts'], "utf-8")
        masks.append(mask)
    return masks


if __name__ == '__main__':
    if INSIDE_BLENDER:
        # Run normally
        main()
    else:
        print('This script is intended to be called from blender like this:')
        print()
        print('blender --background --python render_images.py -- [args]')
        print()
        print('You can also run as a standalone python script to view all')
        print('arguments like this:')
        print()
        print('python render_images.py --help')
