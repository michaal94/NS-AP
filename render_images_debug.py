import os
import sys
import json

PROJECT_PATH = os.path.abspath('.')
sys.path.insert(0, PROJECT_PATH)
PYTHON_PATH = os.popen('which python').read().strip()
if sys.executable.endswith('blender'):
    INSIDE_BLENDER = True
    sys.executable = PYTHON_PATH
else:
    INSIDE_BLENDER = False

from environment.renderer import BlenderRenderer

with open("./environment/blender_default_cfg.json", 'r') as f:
    blender_cfg = json.load(f)

blender_renderer = BlenderRenderer(blender_cfg)

path_scene = "/media/m2data/NS_AP/NS_AP_v1_0_a/weight_order/train/scenes/NS_AP_train_009770.json"
path = "/media/m2data/NS_AP/NS_AP_v1_0_a/weight_order/train/sequences/NS_AP_train_009770/sequence.json"
with open(path, 'r') as f:
    seq = json.load(f)
with open(path_scene, 'r') as f:
    scene_dict = json.load(f)

# print(seq['observations_gt'][12].keys())
# print(scene_dict.keys())

# print(seq['observations_gt'][12]['robot']['robot_body'])

objects = {}

cnt = 0
for o in seq['observations_gt'][1]['objects']:
    if o['file'] in objects:
        objects[o['file'] + f'_{cnt}'] = (o['3d_coords'], o['orientation'])
        cnt += 1
    else:
        objects[o['file']] = (o['3d_coords'], o['orientation'])

blender_renderer.init_scene(scene_dict)
blender_renderer.update_scene(objects, seq['observations_gt'][1]['robot']['robot_body'], seq['observations_gt'][1]['robot']['gripper_body'])

# print(blender_renderer.get_segmentation_masks())

blender_renderer.render('./test/frame_0001.png')