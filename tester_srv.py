from ast import arg
import os
import sys

PROJECT_PATH = os.path.abspath('.')
sys.path.insert(0, PROJECT_PATH)
PYTHON_PATH = os.popen('which python').read().strip()
if sys.executable.endswith('blender'):
    INSIDE_BLENDER = True
    sys.executable = PYTHON_PATH
else:
    INSIDE_BLENDER = False

import json
import argparse
from model.inference import BaselineInferenceTool, InferenceCode
from utils.utils import extract_args

parser = argparse.ArgumentParser()
parser.add_argument(
    '--input_instruction_json', default=''
)
parser.add_argument(
    '--input_instruction_num', default=0, type=int
)
parser.add_argument(
    '--input_scene_dir', default=''
)
parser.add_argument(
    '--output_dir', default=''
)

def main(args):
    assert args.input_scene_dir != ''
    assert args.input_instruction_json != ''
    assert args.output_dir != ''
    instruction_model_params = {
        "name": "Baseline",
        "checkpoint": "./output/checkpoints/instructions.pt",
        "vocab_path": "/home/michal/datasets/NS_AP_v1_0_a/stack/train/vocab.json"
    }
    visual_recognition_model_params = {
        "name": "Baseline",
        "checkpoint": "./output/checkpoints/attributes.pt"
    }
    pose_model_params = {
        "name": "GTLoader"
    }
    action_planner_params = {
        "name": "Baseline",
        "checkpoint": "./output/checkpoints/action.pt"
    }
    segmentation_model_params = {
        "name": "Baseline",
        "checkpoint": "./output/checkpoints/segmentation.pt"
    }

    environment_params = {
        "robots":'Panda',
        "gripper_types":'Robotiq85GripperContinuous',
        "has_offscreen_renderer": False,
        "ignore_done": True,
        "use_camera_obs": False,
        "use_object_obs": True,
        "control_freq": 20,
        "render_collision_mesh": True,
        "renderer":'mujoco',
        "renderer_config":None,
        "bounding_boxes_from_scene":True,
        "return_bboxes":True,
        "debug":True,
        "list_obj_properties":True,
        "has_renderer": False,
        "blender_render": True,
        "output_render_path": f'output/temp',
        "blender_config_path": "./environment/blender_default_cfg.json",
        "controller_config_path": "./environment/controller_cfg.json"
    }

    with open(args.input_instruction_json, 'r') as f:
        instruction_struct = json.load(f)

    sequences_dir = args.output_dir
    os.makedirs(sequences_dir, exist_ok=True)

    idx = args.input_instruction_num
    instr = instruction_struct['instructions'][idx]

    base_fname = os.path.splitext(instr['image_filename'])[0]
    scene_json = os.path.join(args.input_scene_dir, base_fname + '.json')

    save_dir = os.path.join(sequences_dir, base_fname)
    save_path = os.path.join(save_dir, 'frame')
    os.makedirs(save_dir, exist_ok=True)
    environment_params['output_render_path'] = save_path

    with open(scene_json, 'r') as f:
        scene = json.load(f)

    inference_tool = BaselineInferenceTool()
    inference_tool.setup(
        instruction_model_params=instruction_model_params,
        visual_recognition_model_params=visual_recognition_model_params,
        environment_params=environment_params,
        pose_model_params=pose_model_params,
        segmentation_model_params=segmentation_model_params,
        action_planner_params=action_planner_params,
        disable_rendering=False,
        save_dir=save_dir
    )

    inference_tool.load_scene_gt(scene)
    inference_tool.load_instruction(instr)

    outcome = inference_tool.run()
    print("Print Inference Tool exited with the following status:")
    print(InferenceCode.code_to_str(outcome))
    inference_tool.outcome_to_json(outcome)

    with open('./output/results_full_server.json', 'r') as f:
        results = json.load(f)
    
    if base_fname not in results:
        results[base_fname] = outcome
    else:
        if results[base_fname] not in [0, 1]:
            results[base_fname] = outcome
    
    with open('./output/results_full_server.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == '__main__':
    if INSIDE_BLENDER:
        argv = extract_args()
        args = parser.parse_args(argv)
        main(args)
    else:
        args = parser.parse_args()
        print(args)
        main(args)
