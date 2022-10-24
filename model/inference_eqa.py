import os
import json
import copy
import pickle
import time
import numpy as np

import zmq
from PIL import Image

from robosuite import load_controller_config
import robosuite.utils.transform_utils as T

from model import INSTRUCTION_MODELS, VISUAL_RECOGNITION_MODELS
from model import POSE_MODELS, ACTION_PLAN_MODELS

from environment.scene_parser import SceneParser
from environment.tabletop_env_eqa import TabletopEnv
from environment.actions import ActionExecutor

from model.program_executor import ProgramStatus, ProgramExecutor

from utils.utils import CyclicBuffer
from utils.communication import ParamClient, ParamSubscriber

from .ycb_data import COSYPOSE2NAME, COSYPOSE_BBOX, COSYPOSE_TRANSFORM

class InferenceCode:
    CORRECT_ANSWER = 0
    SUCCESSFUL_TASK = 1
    PROGRAM_FAILURE = 2
    INCORRECT_ANSWER = 3
    TIMEOUT = 4
    TASK_FAILURE = 5
    EXECUTION_ERROR = 6
    LOOP_ERROR = 7
    BROKEN_SCENE = 8
    INSTRUCTION_INFERENCE_ERROR = 9
    SCENE_RECOGNITION_ERROR = 10
    PROGRAM_OUTPUT_ERROR = 11
    SCENE_INCONSISTENCY_ERROR = 12

    codebook = [
        "CORRECT_ANSWER",
        "SUCCESSFUL_TASK",
        "PROGRAM_FAILURE",
        "INCORRECT_ANSWER",
        "TIMEOUT",
        "TASK_FAILURE",
        "EXECUTION_ERROR",
        "LOOP_ERROR",
        "BROKEN_SCENE",
        "INSTRUCTION_INFERENCE_ERROR",
        "SCENE_RECOGNITION_ERROR",
        "PROGRAM_OUTPUT_ERROR",
        "SCENE_INCONSISTENCY_ERROR"
    ]

    @staticmethod
    def code_to_str(code):
        return InferenceCode.codebook[code]

class InferenceTool:
    def __init__(self) -> None:
        self.pose_model = None
        self.instruction_model = None
        self.visual_recognition_model = None
        self.scene_gt = None
        self.instruction = None
        # self.scene = None
        self.blender_rendering = False

    def setup(self,
              instruction_model_params={},
              visual_recognition_model_params={},
              pose_model_params={},
              action_planner_params={},
              environment_params={},
              program_executor_params={},
              scene_parser_params={},
              action_executor_params={},
              timeout = 10,
              planning_timeout = 20,
              env_timeout = 1000,
              disable_rendering = False, 
              save_dir = 'temp',
              verbose = True
        ):
        self.verbose = verbose
        self.save_dir = save_dir
        self.disable_rendering = disable_rendering
        self.timeout = timeout
        self.env_timeout = env_timeout
        self.planning_timeout = planning_timeout
        self.environment_params = environment_params
        self._setup_instruction_model(instruction_model_params)
        self._setup_visual_recognition_model(visual_recognition_model_params)
        self._setup_pose_model(pose_model_params)
        self._setup_action_planner(action_planner_params)
        self._setup_visual_recognition_model_gt()
        self._setup_pose_model_gt()
        self._setup_action_planner_gt()
        self.scene_parser = SceneParser(**scene_parser_params)
        self.program_executor = ProgramExecutor(**program_executor_params)
        self.action_executor = ActionExecutor(**action_executor_params)
        self.obs_num = 0
        self.last_render_path = None
        self.loop_detector = CyclicBuffer(2)

    def run(self):
        assert self.instruction_model is not None, "Load instruction to program model (or set GT instruction mode) before running inference"
        assert self.visual_recognition_model is not None, "Load visual recognition model"
        assert self.pose_model is not None, "Load pose estimation model (or set GT pose mode) before running inference"


        # Get program from instruction
        program_list = self.instruction_model.get_program(self.instruction)
        if self.verbose:
            print(f'Program:\t{program_list}')

        # Setup environment
        self._setup_environment()
        # Run some iteration to apply gravity to objects
        action = self.default_environment_action
        _, _, _, _ = self.environment.step(action)
        observation, _, _, _ = self.environment.step(action)
        # input()
        #DEBUG
        self.action_executor.env = self.environment

        # Get first image
        if self.environment.blender_enabled:
            image_path = self.environment.blender_render()
            self.last_render_path = image_path
            image = self._load_image(image_path)
        else:
            # Without blender we don't use the environment to output images
            # Rather for debugging
            # Can be changed though (save images from MuJoCo)
            if not self.disable_rendering:
                self.environment.render()
            image = None

        poses, bboxes = self.pose_model.get_pose(image, observation)
        poses_gt, bboxes_gt = self.pose_model_gt.get_pose(image, observation)
        
        scene_vis = self.visual_recognition_model.get_scene(image, None, self.scene_gt)
        scene_vis_gt = self.visual_recognition_model_gt.get_scene(image, None, self.scene_gt)
        
        self.scene_graph = self._make_scene_graph(scene_vis, poses, bboxes)
        self.scene_graph_gt = self._make_scene_graph(scene_vis_gt, poses_gt, bboxes_gt)

        self.environment.print_robot_configuration()
        # input()

        if self.environment.blender_enabled:
            self.update_sequence_json(observation, ('START', None))

        if not self._check_gt_scene():
            print("Broken scene error")
            return InferenceCode.BROKEN_SCENE

        self.task = self._get_task_from_instruction()
        # print(self.scene_graph)
        # exit()

        for _ in range(self.timeout):
            # self.scene_graph[0]['weight'] = np.array(160)
            # self.scene_graph[1]['weight'] = np.array(140)
            # self.scene_graph[2]['weight'] = np.array(113.6)
            # self.scene_graph[3]['weight'] = np.array(163.4)
            program_output = self.program_executor.execute(self.scene_graph, program_list)
            self.loop_detector.flush()

            print(program_output)
            # print(sce)
            # exit()
            program_status = program_output['STATUS']

            if program_status == ProgramStatus.FAILURE:
                print('Program ended with failure')
                return InferenceCode.PROGRAM_FAILURE

            if program_status == ProgramStatus.SUCCESS:
                print('Program passed through scene graph')
                if self._check_answer(program_output['ANSWER']):
                    print('Answer correct')
                    return InferenceCode.CORRECT_ANSWER
                else:
                    print('Answer incorrect')
                    return InferenceCode.INCORRECT_ANSWER

            if program_status == ProgramStatus.ACTION or program_status == ProgramStatus.FINAL_ACTION:
                planning_tout = self.planning_timeout * len(program_output['ACTION']['target'])
                for _ in range(planning_tout):
                    # print(self.scene_graph[0]['pos'])
                    # print(observation['robot0_eef_pos'])
                    # input()
                    # print(self.scene_graph_gt[2]['in_hand'], self.scene_graph_gt[2]['raised'])
                    # print(self.scene_graph[2]['in_hand'], self.scene_graph[2]['raised'])
                    # self.scene_graph[2]['in_hand'] = True
                    # self.scene_graph[2]['raised'] = True
                    action_plan = self.action_planner.get_action_sequence(
                        program_output['ACTION'],
                        self.scene_graph 
                    )
                    # print(self.scene_graph[1]['in_hand'])
                    # print(self.scene_graph[1]['raised'])
                    # print(self.scene_graph[1]['approached'])
                    # print(self.scene_graph[1]['gripper_over'])
                    # print(self.scene_graph[1]['pos'])
                    # print(self.scene_graph[1]['bbox'])
                    # print(self.scene_graph[2]['in_hand'])
                    # print(self.scene_graph[2]['raised'])
                    # print(self.scene_graph[2]['approached'])
                    # print(self.scene_graph[2]['gripper_over'])
                    # print(self.scene_graph[2]['pos'])
                    # print(self.scene_graph[2]['bbox'])
                    # print(observation['gripper_closed'])
                    # print(observation['gripper_action'])
                    print(action_plan)
                    if self._detect_loop(action_plan):
                        print('Loop detected, exiting')
                        return InferenceCode.LOOP_ERROR
                    self.loop_detector.append(action_plan)
                    # input()
                    # exit()
                    # input()
                    if len(action_plan) == 0:
                        if program_status == ProgramStatus.ACTION:
                            break
                        if self._check_task_completion(self._get_task_from_instruction(), observation):
                            if program_status == ProgramStatus.FINAL_ACTION:
                                print('Correct execution')
                                return InferenceCode.SUCCESSFUL_TASK
                            else:
                                break
                        else:
                            print('Task not reached target')
                            return InferenceCode.TASK_FAILURE
                    action_to_execute = action_plan[0]
                    self.action_executor.set_action(
                        action_to_execute[0], 
                        action_to_execute[1],
                        observation,
                        self.scene_graph
                    )
                    action_executed = False
                    for _ in range(self.env_timeout):
                        if self.action_executor.get_current_action():
                            action = self.action_executor.step(observation)
                        else:
                            action_executed = True
                            break
                        # print(action)
                        observation, _, _, _ = self.environment.step(action)
                        # input()
                        if not self.environment.blender_enabled:
                            if not self.disable_rendering:
                                self.environment.render()
                    if action_executed:
                        if self.environment.blender_enabled:
                            image_path = self.environment.blender_render()
                            self.last_render_path = image_path
                            image = self._load_image(image_path)
                        poses, bboxes = self.pose_model.get_pose(image, observation)
                        poses_gt, bboxes_gt = self.pose_model_gt.get_pose(image, observation)
                        self._update_scene_graph(poses, bboxes, observation)
                        self._update_scene_graph(poses_gt, bboxes_gt, observation, gt=True)
                        if not self._check_gt_scene():
                            print("Broken scene error")
                            return InferenceCode.BROKEN_SCENE
                        if self.environment.blender_enabled:
                            self.update_sequence_json(observation, action_to_execute)
                    else:
                        print('Action not executed correctly')
                        return InferenceCode.EXECUTION_ERROR


        print("Timeout, program execution reiterations exceeded.")
        return InferenceCode.TIMEOUT

        # assert self.pose_model is not None, "Load pose estimation model (or set GT pose mode) before running inference"

        # default_action = self._get_default_action()
        # for i in range(100):
        #     self.environment.step(default_action)
        #     if self.blender_rendering:
        #         self.environment.blender_render()
        #     else:
        #         self.environment.render()

        '''
        + Get program from instruction
        + Setup environment
        + Get first image
        + Get first poses
        + Get visual recognition
        + Associate semantic and geometric graphs
        while before timeout:
            Get program output on the scene graph
            if failure:
                exit
            if success:
                check answer and exit
            if action or action_final:
                for all tasks: 
                    Loop:       
                        if no action happening:
                            Check last reward - check if success 
                            Check last observations (eef pos and ori)
                            Get image
                            Get poses
                            align new poses with old ones (assign to proper graph nodes)
                            update_scene_graph
                            get primitive list
                            set current action in actionexecutor to primitives[0]
                        else:
                            action.step()
                        env.step()
                if action_final:
                    check last status and exit
        '''

    # def _load_scene(self, scene):
    #     assert 'objects' in scene
    #     self.scene = scene

    def _update_scene_graph(self, poses, bboxes, obs, gt=False):
        if gt:
            for i in range(len(self.scene_graph_gt)):
                self.scene_graph_gt[i]['pos'] = poses[i][0]
                self.scene_graph_gt[i]['ori'] = poses[i][1]
                self.scene_graph_gt[i]['bbox'] = bboxes[i]
            eef_pos = obs['robot0_eef_pos']
            eef_ori = obs['robot0_eef_quat']
            gripper_closed = obs['gripper_closed']
            gripper_action = obs['gripper_action']
            # Iterate again cause ultimately there will be matching 
            for idx, obj in enumerate(self.scene_graph_gt):
                obj['in_hand'] = False
                obj['raised'] = False
                obj['approached'] = False
                obj['gripper_over'] = False
                bbox_boundaries = self._check_eef_in_bbox(obj, eef_pos)
                finger_in_bbox = self._check_finger_in_bbox(obj, eef_pos, eef_ori)
                # print(obj['name'], finger_in_bbox)
                # print(bbox_boundaries)
                # print(gripper_closed)
                # print(gripper_action)
                if len(bbox_boundaries) == 3 or finger_in_bbox:
                    if gripper_closed and gripper_action > -0.99:
                        obj['in_hand'] = True
                    else:
                        obj['approached'] = True
                elif ('x' in bbox_boundaries and 'y' in bbox_boundaries):
                    obj['gripper_over'] = True
                if all([obj['bbox'][i][2] > 0.05 for i in range(8)]):
                    obj['raised'] = True
                    if obs['grasped_obj_idx'] == idx:
                        obj['weight'] = obs['weight_measurement']
        else:
            #TODO
            # Do the pose matching w.r.t history
            for i in range(len(self.scene_graph)):
                self.scene_graph[i]['pos'] = poses[i][0]
                self.scene_graph[i]['ori'] = poses[i][1]
                self.scene_graph[i]['bbox'] = bboxes[i]
            eef_pos = obs['robot0_eef_pos']
            eef_ori = obs['robot0_eef_quat']
            gripper_closed = obs['gripper_closed']
            gripper_action = obs['gripper_action']
            # print(obs)
            # Iterate again cause ultimately there will be matching 
            for obj in self.scene_graph:
                obj['in_hand'] = False
                obj['raised'] = False
                obj['approached'] = False
                obj['gripper_over'] = False
                bbox_boundaries = self._check_eef_in_bbox(obj, eef_pos)
                finger_in_bbox = self._check_finger_in_bbox(obj, eef_pos, eef_ori)
                # print(bbox_boundaries)
                # print(gripper_closed)
                # print(gripper_action)
                if len(bbox_boundaries) == 3 or finger_in_bbox:
                    if gripper_closed and gripper_action > -0.99:
                        obj['in_hand'] = True
                    else:
                        obj['approached'] = True
                elif ('x' in bbox_boundaries and 'y' in bbox_boundaries):
                    obj['gripper_over'] = True
                if all([obj['bbox'][i][2] > 0.05 for i in range(8)]):
                    obj['raised'] = True
                if obj['raised'] and obj['in_hand']:
                    obj['weight'] = obs['weight_measurement']
                # print(obj['name'], obj['in_hand'], obj['raised'])

    def _check_eef_in_bbox(self, obj, eef_pos):
        bbox = obj['bbox']
        x_vec = bbox[0] - bbox[1]
        y_vec = bbox[0] - bbox[3]
        z_vec = bbox[4] - bbox[0]
        x_size = np.linalg.norm(x_vec)
        y_size = np.linalg.norm(y_vec)
        z_size = np.linalg.norm(z_vec)
        # print(bbox.shape)
        bbox_mid = np.mean(bbox, axis=0)
        mid_to_eef_vec = eef_pos - bbox_mid
        axes_in = []
        x_proj = np.abs(np.dot(mid_to_eef_vec, x_vec) / x_size)
        if x_proj < x_size / 2:
            axes_in.append('x')
        y_proj = np.abs(np.dot(mid_to_eef_vec, y_vec) / y_size)
        if y_proj < y_size / 2:
            axes_in.append('y')
        z_proj = np.abs(np.dot(mid_to_eef_vec, z_vec) / z_size)
        if z_proj < z_size / 2:
            axes_in.append('z')
        return axes_in 

    def _check_finger_in_bbox(self, obj, eef_pos, eef_ori):
        finger_dist = self.action_executor.gripper_open_width
        gripper_y_dir = T.quat2mat(eef_ori)[:, 1]
        gripper_y_dir = gripper_y_dir / np.linalg.norm(gripper_y_dir)
        finger1 = eef_pos + finger_dist * gripper_y_dir
        finger2 = eef_pos - finger_dist * gripper_y_dir
        if len(self._check_eef_in_bbox(obj, finger1)) == 3:
            return True
        if len(self._check_eef_in_bbox(obj, finger2)) == 3:
            return True
        return False


    def _check_task_completion(self, action, obs):
        if action['task'] == 'measure_weight':
            if self.scene_graph_gt[action['target'][0][0]]['raised']:
                if obs['weight_measurement'] > 0:
                    self.scene_graph_gt[action['target'][0][0]]['weight'] = obs['weight_measurement']
                    return True
        elif action['task'] == 'pick_up':
            print(action['target'][0][0])
            if self.scene_graph_gt[action['target'][0][0]]['raised']:
                 return True
        elif action['task'] == 'stack':
            return self._check_stack(action['target'][0])
        elif 'move' in action['task']:
            return self._check_move(action)
        return False

    def _check_move(self, task):
        side = task['task'].split('[')[1].strip(']')
        for idx in task['target'][0]:
            if self.scene_graph_gt[idx]['in_hand']:
                return False
            y_pos = self.scene_graph_gt[idx]['pos'][1]
            if (side == 'right' and y_pos < 0):
                return False
            if (side == 'left' and y_pos > 0):
                return False
        return True

    def _check_stack(self, targets):
        targets = targets[0]
        for i in reversed(range(len(targets) - 1)):
            top = targets[i]
            bottom = targets[i + 1]
            pos_top = self.scene_graph_gt[top]['pos']
            pos_bot = self.scene_graph_gt[bottom]['pos']
            if (pos_top[2] - pos_bot[2]) < 0.0:
                return False
            if np.linalg.norm(pos_top[0:2] - pos_bot[0:2]) > 0.1:
                return False
        return True

    def _check_answer(self, answer):
        gt_answer = self.instruction['program'][-1]['output']
        if gt_answer is None:
            return False
        if answer is None:
            return False
        # print(answer)
        if isinstance(answer[0], list):
            if not isinstance(gt_answer[0], list):
                # answer = [a[0] for a in answer]
                answer = answer[0]
                # print(answer)
                if type(answer[0]).__module__ == np.__name__:
                    answer = [a.tolist() for a in answer]
        # print(gt_answer, answer)
        
        if len(answer) != len(gt_answer):
            return False
        for a in answer:
            if a not in gt_answer:
                return False
        return True

    def _make_scene_graph(self, scene, poses, bboxes):
        scene_graph = []
        for i, o in enumerate(scene):
            scene_graph.append(copy.deepcopy(o))
            scene_graph[-1]['pos'] = poses[i][0]
            scene_graph[-1]['ori'] = poses[i][1]
            scene_graph[-1]['bbox'] = bboxes[i]
            scene_graph[-1]['in_hand'] = False
            scene_graph[-1]['raised'] = False
            scene_graph[-1]['approached'] = False
            scene_graph[-1]['gripper_over'] = False
        return scene_graph

    def load_scene_gt(self, scene):
        assert 'objects' in scene
        self.scene_gt = scene
    
    def load_instruction(self, instruction_dict):
        assert 'instruction' in instruction_dict
        self.instruction = instruction_dict

    def _setup_instruction_model(self, params):
        assert params['name'] in INSTRUCTION_MODELS, "Unknown instruction model"
        self.instruction_model = INSTRUCTION_MODELS[params['name']]()

    def _setup_visual_recognition_model(self, params):
        assert params['name'] in VISUAL_RECOGNITION_MODELS, "Unknown visual model"
        self.visual_recognition_model = VISUAL_RECOGNITION_MODELS[params['name']]()

    def _setup_pose_model(self, params):
        assert params['name'] in POSE_MODELS, "Unknown pose model"
        self.pose_model = POSE_MODELS[params['name']]()

    def _setup_action_planner(self, params):
        assert params['name'] in ACTION_PLAN_MODELS, "Unknown action planner"
        self.action_planner = ACTION_PLAN_MODELS[params['name']]()

    def _setup_visual_recognition_model_gt(self):
        self.visual_recognition_model_gt = VISUAL_RECOGNITION_MODELS["GTLoader"]()

    def _setup_pose_model_gt(self):
        self.pose_model_gt = POSE_MODELS["GTLoader"]()

    def _setup_action_planner_gt(self):
        self.action_planner_gt = ACTION_PLAN_MODELS["GTLoader"]()

    def _setup_environment(self):
        assert self.scene_gt is not None, "Load GT scene to initialise simulation"
        controller_cfg_path = self.environment_params.pop('controller_config_path', None)
        if controller_cfg_path is not None:
            controller_cfg = load_controller_config(controller_cfg_path)
        else:
            controller_cfg = None
        self.environment_params['controller_configs'] = controller_cfg
        blender_cfg_path = self.environment_params.pop('blender_config_path', None)
        if blender_cfg_path is not None:
            with open(blender_cfg_path, 'r') as f:
                blender_cfg = json.load(f)
        else:
            blender_cfg = None
        self.environment_params['blender_config'] = blender_cfg
        self.environment_params['scene_dict'] = self.scene_gt
        obj_dict = self.scene_parser.parse_scene(self.scene_gt)
        self.environment_params['objs'] = obj_dict

        self.environment = TabletopEnv(**self.environment_params)
        if 'blender_render' in self.environment_params:
            self.blender_rendering = self.environment_params['blender_render']
        else:
            self.blender_rendering = False

        action_dim = self.environment.action_dim
        self.default_environment_action = np.array(
            (action_dim - 1) * [0] + [-1]
        )

    def _load_image(self, path):
        return Image.open(path)

    def _get_task_from_instruction(self):
        prog_last = self.instruction["program"][-1]
        outp = prog_last["output"]
        if isinstance(prog_last["output"][0], list):
            new_outp = []
            for o_item in prog_last["output"]:
                new_outp += o_item
            outp = [new_outp]
        if 'move' in prog_last["type"]:
            out_task = f"{prog_last['type']}[{prog_last['input_value']}]"
        else:
            out_task = prog_last["type"]
        task = {
            'task': out_task,
            'target': [outp]
        }
        return task

    def update_sequence_json(self, obs, last_action=None):
        if not self.environment.blender_enabled:
            return
        if self.obs_num == 0:
            json_name = "sequence.json"
            self.json_path = os.path.join(self.save_dir, json_name)
            info_struct = {
                "info": {
                    "image_filename": self.instruction["image_filename"],
                    "instruction": self.instruction["instruction"],
                    "task": self._get_task_from_instruction()
                },
                "observations": [],
                "observations_gt": [],
                "image_paths": [],
                "result": None
            }
            with open(self.json_path, 'w') as f:
                json.dump(info_struct, f, indent=4)

        with open(self.json_path, 'r') as f:
            info_struct = json.load(f)

        info_struct['image_paths'].append(self.last_render_path)
        obs_set = {}
        obs_set['robot'] = {
            'pos': obs['robot0_eef_pos'].tolist(),
            'ori': obs['robot0_eef_quat'].tolist(),
            'gripper_action': obs['gripper_action'].tolist(),
            'gripper_closed': obs['gripper_closed'].tolist(),
            'weight_measurement': obs['weight_measurement'].tolist(),
            'action': last_action
        }

        obs_set['objects'] = []
        for obj in self.scene_graph:
            obs_set['objects'].append(copy.deepcopy(obj))
            obs_set['objects'][-1]['bbox'] = obs_set['objects'][-1]['bbox'].tolist()
            obs_set['objects'][-1]['pos'] = obs_set['objects'][-1]['pos'].tolist()
            obs_set['objects'][-1]['ori'] = obs_set['objects'][-1]['ori'].tolist()
            if obs_set['objects'][-1]['weight'] is not None:
                obs_set['objects'][-1]['weight'] = obs_set['objects'][-1]['weight'].tolist()

        obs_set_gt = {}
        robot_body, gripper_body = self.environment.get_robot_configuration()
        for k, v in robot_body.items():
            robot_body[k] = (v[0].tolist(), v[1].tolist())
        for k, v in gripper_body.items():
            gripper_body[k] = (v[0].tolist(), v[1].tolist())

        masks = self.environment.get_segmentation_masks()
        obs_set_gt['robot'] = {
            'pos': obs['robot0_eef_pos'].tolist(),
            'ori': obs['robot0_eef_quat'].tolist(),
            'gripper_action': obs['gripper_action'].tolist(),
            'gripper_closed': obs['gripper_closed'].tolist(),
            'weight_measurement': obs['weight_measurement'].tolist(),
            'grasped_obj_idx': obs['grasped_obj_idx'].tolist(),
            'robot_body': robot_body,
            'gripper_body': gripper_body,
            'robot_mask': masks['robot'],
            'table_mask': masks['table']
        }

        obs_set_gt['objects'] = []
        for i, obj in enumerate(self.scene_graph_gt):
            obs_set_gt['objects'].append(copy.deepcopy(obj))
            obs_set_gt['objects'][i]['mask'] = masks['objects'][i]
            obs_set_gt['objects'][-1]['bbox'] = obs_set_gt['objects'][-1]['bbox'].tolist()
            obs_set_gt['objects'][-1]['pos'] = obs_set_gt['objects'][-1]['pos'].tolist()
            obs_set_gt['objects'][-1]['ori'] = obs_set_gt['objects'][-1]['ori'].tolist()
            if obs_set_gt['objects'][-1]['weight'] is not None:
                obs_set_gt['objects'][-1]['weight'] = obs_set_gt['objects'][-1]['weight'].tolist()


        info_struct['observations'].append(obs_set)
        info_struct['observations_gt'].append(obs_set_gt)
        with open(self.json_path, 'w') as f:
            json.dump(info_struct, f, indent=4)
        self.obs_num += 1

    def outcome_to_json(self, outcome):
        if not self.environment.blender_enabled:
            return
        with open(self.json_path, 'r') as f:
            info_struct = json.load(f)

        info_struct['result'] = outcome

        with open(self.json_path, 'w') as f:
            json.dump(info_struct, f, indent=4)

    def _detect_loop(self, target_list):
        if len(self.loop_detector) < 2:
            return False
        last_commands = self.loop_detector.get()
        for source_list in last_commands:
            if len(source_list) != len(target_list):
                continue
            else:
                same = True
                for i, command in enumerate(source_list):
                    if command != target_list[i]:
                        same = False
                        break
                if same:
                    return True
        return False

    def _check_gt_scene(self):
        for i in range(len(self.scene_graph_gt)):
            if self.scene_graph_gt[i]['pos'][2] < -0.1:
                return False
        return True


class BaselineInferenceTool:
    def __init__(self) -> None:
        self.pose_model = None
        self.instruction_model = None
        self.visual_recognition_model = None
        self.scene_gt = None
        self.instruction = None
        # self.scene = None
        self.blender_rendering = False

    def setup(self,
              instruction_model_params={},
              visual_recognition_model_params={},
              pose_model_params={},
              action_planner_params={},
              environment_params={},
              program_executor_params={},
              scene_parser_params={},
              action_executor_params={},
              segmentation_model_params={},
              timeout = 10,
              planning_timeout = 20,
              env_timeout = 1000,
              disable_rendering = False, 
              save_dir = 'temp',
              verbose = True
        ):
        self.verbose = verbose
        self.save_dir = save_dir
        self.disable_rendering = disable_rendering
        self.timeout = timeout
        self.env_timeout = env_timeout
        self.planning_timeout = planning_timeout
        self.environment_params = environment_params
        self._setup_instruction_model(instruction_model_params)
        self._setup_instruction_model_gt()
        self._setup_visual_recognition_model(visual_recognition_model_params)
        # self._setup_action_planner(action_planner_params)
        self._setup_visual_recognition_model_gt()
        self._setup_pose_model_gt()
        self._setup_action_planner_gt()
        self._setup_action_planner(action_planner_params)
        self._setup_segmentation_model(segmentation_model_params)
        self.scene_parser = SceneParser(**scene_parser_params)
        self.program_executor = ProgramExecutor(**program_executor_params)
        self.action_executor = ActionExecutor(**action_executor_params)
        self.obs_num = 0
        self.last_render_path = None
        self.loop_detector = CyclicBuffer(2)
        self.json_path = None

    def run(self):
        assert self.instruction_model is not None, "Load instruction to program model (or set GT instruction mode) before running inference"
        # assert self.visual_recognition_model is not None, "Load visual recognition model"
        # assert self.pose_model is not None, "Load pose estimation model (or set GT pose mode) before running inference"


        # Get program from instruction
        program_list = self.instruction_model.get_program(self.instruction)
        program_list_gt = self.instruction_model_gt.get_program(self.instruction)
        if self.verbose:
            print(f'Program:\t{program_list}')

        if not self._check_program(program_list, program_list_gt):
            print("Incorrect instruction inference")
            return InferenceCode.INSTRUCTION_INFERENCE_ERROR

        # Setup environment
        self._setup_environment()
        # Run some iteration to apply gravity to objects
        action = self.default_environment_action
        _, _, _, _ = self.environment.step(action)
        observation, _, _, _ = self.environment.step(action)
        # input()
        #DEBUG
        self.action_executor.env = self.environment

        # Get first image
        if self.environment.blender_enabled:
            image_path = self.environment.blender_render()
            self.last_render_path = image_path
            image = self._load_image(image_path).convert("RGB")
        else:
            # Without blender we don't use the environment to output images
            # Rather for debugging
            # Can be changed though (save images from MuJoCo)
            if not self.disable_rendering:
                self.environment.render()
            image = None

        segmentation_masks, labels_org = self.segmentation_model.get_segmenation(image)
        if len(segmentation_masks) != len(self.scene_gt['objects']):
            print("Incorrect initial scene recognition")
            return InferenceCode.SCENE_RECOGNITION_ERROR

        poses_gt, bboxes_gt = self.pose_model_gt.get_pose(image, observation)
        scene_vis_gt = self.visual_recognition_model_gt.get_scene(image, None, self.scene_gt)
        poses, bboxes_local, scene_vis = self.visual_recognition_model.get_scene_pose(
            image, segmentation_masks, self.scene_gt, return_bboxes=True
        )

        poses, bboxes_local, scene_vis = self._align_to_gt(
            poses, bboxes_local, scene_vis
        )

        poses = self._correct_poses(poses)
        bboxes = self._boxes_to_global(bboxes_local, poses)

        # for i in range(len(bboxes)):
        #     print(bboxes[i])
        #     print(bboxes_gt[i])
        #     print()
        # exit()
        if scene_vis is None:
            print("Incorrect initial scene recognition")
            return InferenceCode.SCENE_RECOGNITION_ERROR

        self.scene_graph = self._make_scene_graph(scene_vis, poses, bboxes)
        self.scene_graph_gt = self._make_scene_graph(scene_vis_gt, poses_gt, bboxes_gt)

        if self.environment.blender_enabled:
            self.update_sequence_json(observation, ('START', None))

        if not self._check_gt_scene():
            print("Broken scene error")
            return InferenceCode.BROKEN_SCENE

        self.task = self._get_task_from_instruction()
        # print(self.scene_graph)
        previous_pick_up_target = None
        prev_action = None
        current_target = None

        for _ in range(self.timeout):
            # print([o['colour'] for o in self.scene_graph])
            # print([o['colour'] for o in self.scene_graph_gt])
            # print(program_list)
            # print(program_list_gt)
            program_output = self.program_executor.execute(self.scene_graph, program_list)
            program_output_gt = self.program_executor.execute(self.scene_graph_gt, program_list_gt)
            self.loop_detector.flush()

            print(program_output)
            print(program_output_gt)
            # print(self.scene_graph[4])
            if not self._compare_program_outputs(program_output, program_output_gt):
                print("Incorrect program output")
                return InferenceCode.PROGRAM_OUTPUT_ERROR
            # print(sce)
            program_status = program_output['STATUS']

            if program_status == ProgramStatus.FAILURE:
                print('Program ended with failure')
                return InferenceCode.PROGRAM_FAILURE

            if program_status == ProgramStatus.SUCCESS:
                print('Program passed through scene graph')
                if self._check_answer(program_output['ANSWER']):
                    print('Answer correct')
                    return InferenceCode.CORRECT_ANSWER
                else:
                    print('Answer incorrect')
                    return InferenceCode.INCORRECT_ANSWER

            if program_status == ProgramStatus.ACTION or program_status == ProgramStatus.FINAL_ACTION:
                planning_tout = self.planning_timeout * len(program_output['ACTION']['target'])
                all_targets = None
                if len(program_output['ACTION']['target'][0]) > 2 and program_output['ACTION']['task'] == 'stack':
                    all_targets = program_output['ACTION']['target'][0]
                    current_target = len(all_targets) - 2
                for _ in range(planning_tout):
                    if current_target is not None:
                        if current_target < 0:
                            return InferenceCode.INSTRUCTION_INFERENCE_ERROR
                        program_output['ACTION']['target'][0] = all_targets[current_target:current_target+2]
                    # print(program_output['ACTION'])
                    action_plan_gt = self.action_planner_gt.get_action_sequence(
                        program_output_gt['ACTION'],
                        self.scene_graph_gt
                    )
                    action_plan = self.action_planner.get_action_sequence(
                        program_output['ACTION'],
                        self.scene_graph 
                    )
                    print(action_plan)
                    print(action_plan_gt)
                    if prev_action == 'put_down':
                        action_plan = [('release', None)]
                    if prev_action == 'pick_up':
                        if program_output['ACTION']['task'] == 'move_to[right]':
                            action_plan = [('move', 'right')]
                        if program_output['ACTION']['task'] == 'move_to[left]':
                            action_plan = [('move', 'left')]
                    prev_action = action_plan[0][0]
                    if self._detect_loop(action_plan):
                        print('Loop detected, exiting')
                        return InferenceCode.LOOP_ERROR
                    self.loop_detector.append(action_plan)
                    if len(action_plan_gt) == 0:
                        if program_status == ProgramStatus.ACTION:
                            break
                        if self._check_task_completion(self._get_task_from_instruction(), observation):
                            if program_status == ProgramStatus.FINAL_ACTION:
                                print('Correct execution')
                                return InferenceCode.SUCCESSFUL_TASK
                            else:
                                break
                        else:
                            print('Task not reached target')
                            return InferenceCode.TASK_FAILURE
                    action_to_execute = action_plan[0]
                    if current_target is not None:
                        if action_to_execute[0] == 'release':
                            current_target -= 1
                    if action_to_execute[0] in ['put_down']:
                        if previous_pick_up_target is not None:
                            action_to_execute = (action_to_execute[0], previous_pick_up_target)
                            # action_to_execute[1] = previous_pick_up_target
                            previous_pick_up_target = None
                    if action_to_execute[0] in ['pick_up']:
                        previous_pick_up_target = action_to_execute[1]
                    self.action_executor.set_action(
                        action_to_execute[0], 
                        action_to_execute[1],
                        observation,
                        self.scene_graph
                    )
                    action_executed = False
                    for _ in range(self.env_timeout):
                        if self.action_executor.get_current_action():
                            action = self.action_executor.step(observation)
                        else:
                            action_executed = True
                            break
                        observation, _, _, _ = self.environment.step(action)
                        # input()
                        if not self.environment.blender_enabled:
                            if not self.disable_rendering:
                                self.environment.render()
                    # self.environment.blender_render()
                    if action_executed:
                        if self.environment.blender_enabled:
                            image_path = self.environment.blender_render()
                            self.last_render_path = image_path
                            image = self._load_image(image_path).convert("RGB")

                        segmentation_masks, labels = self.segmentation_model.get_segmenation(image)
                        if len(labels) > len(labels_org):
                            labels = labels[0:len(labels_org)]
                            segmentation_masks = segmentation_masks[0:len(labels_org)]
                            if sorted(labels) != sorted(labels_org):
                                print('Additional objects detected')
                                return InferenceCode.SCENE_INCONSISTENCY_ERROR
                        
                        # import torchvision.transforms as Trr
                        # for i, m in enumerate(segmentation_masks):
                        #     print(m.sum())
                        #     Trr.ToPILImage()(m).save(f'./test/m_{i}.png')
                        poses_gt, bboxes_gt = self.pose_model_gt.get_pose(image, observation)
                        
                        poses, _, scene_vis = self.visual_recognition_model.get_scene_pose(
                            image, segmentation_masks, self.scene_gt, return_bboxes=False
                        )
                        # poses, bboxes = self.pose_model.get_pose(image, observation)
                        
                            
                        poses = self._correct_poses(poses)

                        poses, bboxes = self._align_to_past(
                            poses, bboxes, scene_vis
                        )
                        poses = self._correct_height(poses, poses_gt)
                        bboxes = self._boxes_to_global(bboxes_local, poses)


                        self._update_scene_graph(poses, bboxes, observation)
                        self._update_scene_graph(poses_gt, bboxes_gt, observation, gt=True)
                        if not self._check_gt_scene():
                            print("Broken scene error")
                            return InferenceCode.BROKEN_SCENE
                        if self.environment.blender_enabled:
                            self.update_sequence_json(observation, action_to_execute)
                    else:
                        print('Action not executed correctly')
                        return InferenceCode.EXECUTION_ERROR


        print("Timeout, program execution reiterations exceeded.")
        return InferenceCode.TIMEOUT

    def _check_program(self, prog, gt):
        if len(prog) != len(gt):
            return False
        if prog != gt:
            return False
        return True

    def _align_to_gt(self, poses, bboxes, scene):
        new_poses = [None] * len(poses)
        new_bboxes = [None] * len(bboxes)
        new_scene = [None] * len(scene)
        positions = [np.array(o['3d_coords']) for o in self.scene_gt['objects']]
        new_idxs = []
        for obj in scene:
            diffs = [np.abs(obj['3d_coords'] - p).sum() for p in positions]
            min_idx = diffs.index(min(diffs))
            new_idxs.append(min_idx)
        if sorted(new_idxs) != list(range(len(self.scene_gt['objects']))):
            return None, None, None
        for i, idx in enumerate(new_idxs):
            new_poses[idx] = poses[i]
            new_bboxes[idx] = bboxes[i]
            new_scene[idx] = scene[i]
        return new_poses, new_bboxes, new_scene

    def _align_to_past(self, poses, bboxes, scene):
        new_poses = [None] * len(self.scene_graph)
        new_bboxes = [None] * len(self.scene_graph)
        new_idxs = [None] * len(scene)
        old_idxs = list(range(len(self.scene_graph)))

        for i in range(len(scene)):
            found_obj = scene[i]
            matches = []
            for old_idx, old_obj in enumerate(self.scene_graph):
                if old_obj['name'] == found_obj['name']:
                    if old_obj['material'] == found_obj['material']:
                        if old_obj['shape'] == found_obj['shape']:
                            if old_obj['colour'] == found_obj['colour']:
                                matches.append(old_idx)
            if len(matches) == 1:
                new_idxs[i] = matches[0]
                continue

        for i in range(len(new_idxs)):
            if new_idxs[i] is None:
                continue
            for j in range(i, len(new_idxs)):
                if i == j:
                    continue
                if new_idxs[i] == new_idxs[j]:
                    new_idxs[i] = None
                    new_idxs[j] = None

        for i in range(len(scene)):
            if new_idxs[i] is not None:
                continue
            found_pose = poses[i]
            found_pos = found_pose[0]
            found_ori = found_pose[1]
            distances = []
            distances_xy = []
            old_heights = []
            for obj in self.scene_graph:
                old_pos = obj['pos']
                old_heights.append(old_pos[2])
                distances.append(
                    np.linalg.norm(
                        np.array(found_pos) - np.array(old_pos)
                    )
                )
                distances_xy.append(
                    np.linalg.norm(
                        np.array(found_pos[0:2]) - np.array(old_pos[0:2])
                    )
                )
            min_dist_idx = distances.index(min(distances))
            if distances[min_dist_idx] < 0.02:
                if min_dist_idx not in new_idxs:
                    new_idxs[i] = min_dist_idx
                    continue

            if found_pos[2] > 0.05:
                # If it is in the air and sth else was in the air
                # it must be the same
                max_old_height_idx = old_heights.index(max(old_heights))
                if old_heights[max_old_height_idx] > 0.05:
                    if max_old_height_idx not in new_idxs:
                        new_idxs[i] = max_old_height_idx
                        continue

                min_dist_idx = distances.index(min(distances_xy))
                if distances_xy[min_dist_idx] < 0.04:
                    if min_dist_idx not in new_idxs:
                        new_idxs[i] = min_dist_idx
                        continue

        for i in range(len(scene)):
            if new_idxs[i] is not None:
                continue
            found_pose = poses[i]
            found_pos = found_pose[0]
            distances_xy = []
            for obj in self.scene_graph:
                old_pos = obj['pos']
                distances_xy.append(
                    np.linalg.norm(
                        np.array(found_pos[0:2]) - np.array(old_pos[0:2])
                    )
                )
            indices_increasing = sorted(range(len(distances_xy)), key=lambda k: distances_xy[k])
            for idx in indices_increasing:
                if idx not in new_idxs:
                    new_idxs[i] = min_dist_idx
                    continue
            
        for i, idx in enumerate(new_idxs):
            if idx is not None:
                new_poses[idx] = poses[i]
                new_bboxes[idx] = bboxes[i]

        for i in range(len(new_poses)):
            if new_poses[i] is None:
                new_poses[i] = (
                    self.scene_graph[i]['pos'],
                    self.scene_graph[i]['ori']
                )
                new_bboxes[i] = self.scene_graph[i]['bbox']

        return new_poses, new_bboxes

    def _correct_poses(self, poses):
        new_poses = []

        for i, (pos, ori) in enumerate(poses):
            gt_ori = np.array(self.scene_gt['objects'][i]['orientation'])
            if np.abs(gt_ori + ori).sum() < np.abs(ori - gt_ori).sum():
                ori = -ori
            # ori = -ori
            # print(ori, gt_ori)
            # print(T.mat2quat(T.quat2mat(ori)), gt_ori)
            # print(T.convert_quat(T.mat2quat(T.quat2mat(ori)), to="xyzw"))
            # print(T.convert_quat(gt_ori, to="xyzw"))
            # print()
            ori_new = T.convert_quat(T.mat2quat(T.quat2mat(ori)), to="xyzw")
            p = pos
            if pos[2] < 0:
                pos[2] = 0
            new_poses.append(
                (pos, ori_new)
            )
        return new_poses
    
    def _compare_program_outputs(self, prog, prog_gt):
        if prog['STATUS'] != prog_gt['STATUS']:
            return False
        if prog_gt['ACTION'] is not None:
            if prog['ACTION'] is None:
                return False
            if prog['ACTION']['task'] != prog_gt['ACTION']['task']:
                return False
        return True

    def _boxes_to_global(self, boxes_local, poses):
        boxes_global = []
        for i, b_local in enumerate(boxes_local):
            pos, ori = poses[i]
            bbox_local = np.concatenate(
                (
                    b_local.T,
                    np.ones((b_local.shape[0], 1)).T
                )
            )
            obj_world_mat = T.pose2mat((pos, ori))
            bbox_world = np.matmul(obj_world_mat, bbox_local)[:-1, :].T
            boxes_global.append(bbox_world)
        return boxes_global

    def _update_scene_graph(self, poses, bboxes, obs, gt=False):
        if gt:
            for i in range(len(self.scene_graph_gt)):
                self.scene_graph_gt[i]['pos'] = poses[i][0]
                self.scene_graph_gt[i]['ori'] = poses[i][1]
                self.scene_graph_gt[i]['bbox'] = bboxes[i]
            eef_pos = obs['robot0_eef_pos']
            eef_ori = obs['robot0_eef_quat']
            gripper_closed = obs['gripper_closed']
            gripper_action = obs['gripper_action']
            # Iterate again cause ultimately there will be matching 
            for idx, obj in enumerate(self.scene_graph_gt):
                obj['in_hand'] = False
                obj['raised'] = False
                obj['approached'] = False
                obj['gripper_over'] = False
                bbox_boundaries = self._check_eef_in_bbox(obj, eef_pos)
                finger_in_bbox = self._check_finger_in_bbox(obj, eef_pos, eef_ori)
                if len(bbox_boundaries) == 3 or finger_in_bbox:
                    if gripper_closed and gripper_action > -0.99:
                        obj['in_hand'] = True
                    else:
                        obj['approached'] = True
                elif ('x' in bbox_boundaries and 'y' in bbox_boundaries):
                    obj['gripper_over'] = True
                if all([obj['bbox'][i][2] > 0.05 for i in range(8)]):
                    obj['raised'] = True
                    if obs['grasped_obj_idx'] == idx:
                        obj['weight'] = obs['weight_measurement']
        else:
            for i in range(len(self.scene_graph)):
                self.scene_graph[i]['pos'] = poses[i][0]
                self.scene_graph[i]['ori'] = poses[i][1]
                self.scene_graph[i]['bbox'] = bboxes[i]
            eef_pos = obs['robot0_eef_pos']
            eef_ori = obs['robot0_eef_quat']
            gripper_closed = obs['gripper_closed']
            gripper_action = obs['gripper_action']
            # print(obs)
            # Iterate again cause ultimately there will be matching 
            for obj in self.scene_graph:
                obj['in_hand'] = False
                obj['raised'] = False
                obj['approached'] = False
                obj['gripper_over'] = False
                bbox_boundaries = self._check_eef_in_bbox(obj, eef_pos)
                finger_in_bbox = self._check_finger_in_bbox(obj, eef_pos, eef_ori)
                # print(bbox_boundaries)
                # print(gripper_closed)
                # print(gripper_action)
                if len(bbox_boundaries) == 3 or finger_in_bbox:
                    if gripper_closed and gripper_action > -0.99:
                        obj['in_hand'] = True
                    else:
                        obj['approached'] = True
                elif ('x' in bbox_boundaries and 'y' in bbox_boundaries):
                    obj['gripper_over'] = True
                if all([obj['bbox'][i][2] > 0.05 for i in range(8)]):
                    obj['raised'] = True
                if obj['raised'] and obj['in_hand']:
                    obj['weight'] = obs['weight_measurement']
                # print(obj['name'], obj['in_hand'], obj['raised'])
            if obs['weight_measurement'] > 0:
                if not any([o['in_hand'] for o in self.scene_graph]):
                    distances = [np.linalg.norm(eef_pos - o['pos']) for o in self.scene_graph]
                    min_idx = distances.index(min(distances))
                    self.scene_graph[min_idx]['gripper_over'] = False
                    self.scene_graph[min_idx]['approached'] = False
                    self.scene_graph[min_idx]['in_hand'] = True
                    self.scene_graph[min_idx]['weight'] = obs['weight_measurement']


    def _check_eef_in_bbox(self, obj, eef_pos):
        bbox = obj['bbox']
        x_vec = bbox[0] - bbox[1]
        y_vec = bbox[0] - bbox[3]
        z_vec = bbox[4] - bbox[0]
        x_size = np.linalg.norm(x_vec)
        y_size = np.linalg.norm(y_vec)
        z_size = np.linalg.norm(z_vec)
        # print(bbox.shape)
        bbox_mid = np.mean(bbox, axis=0)
        mid_to_eef_vec = eef_pos - bbox_mid
        axes_in = []
        x_proj = np.abs(np.dot(mid_to_eef_vec, x_vec) / x_size)
        if x_proj < x_size / 2:
            axes_in.append('x')
        y_proj = np.abs(np.dot(mid_to_eef_vec, y_vec) / y_size)
        if y_proj < y_size / 2:
            axes_in.append('y')
        z_proj = np.abs(np.dot(mid_to_eef_vec, z_vec) / z_size)
        if z_proj < z_size / 2:
            axes_in.append('z')
        return axes_in 

    def _check_finger_in_bbox(self, obj, eef_pos, eef_ori):
        finger_dist = self.action_executor.gripper_open_width
        gripper_y_dir = T.quat2mat(eef_ori)[:, 1]
        gripper_y_dir = gripper_y_dir / np.linalg.norm(gripper_y_dir)
        finger1 = eef_pos + finger_dist * gripper_y_dir
        finger2 = eef_pos - finger_dist * gripper_y_dir
        if len(self._check_eef_in_bbox(obj, finger1)) == 3:
            return True
        if len(self._check_eef_in_bbox(obj, finger2)) == 3:
            return True
        return False


    def _check_task_completion(self, action, obs):
        if action['task'] == 'measure_weight':
            if self.scene_graph_gt[action['target'][0][0]]['raised']:
                if obs['weight_measurement'] > 0:
                    self.scene_graph_gt[action['target'][0][0]]['weight'] = obs['weight_measurement']
                    return True
        elif action['task'] == 'pick_up':
            print(action['target'][0][0])
            if self.scene_graph_gt[action['target'][0][0]]['raised']:
                 return True
        elif action['task'] == 'stack':
            return self._check_stack(action['target'][0])
        elif 'move' in action['task']:
            return self._check_move(action)
        return False

    def _check_move(self, task):
        side = task['task'].split('[')[1].strip(']')
        for idx in task['target'][0]:
            if self.scene_graph_gt[idx]['in_hand']:
                return False
            y_pos = self.scene_graph_gt[idx]['pos'][1]
            if (side == 'right' and y_pos < 0):
                return False
            if (side == 'left' and y_pos > 0):
                return False
        return True

    def _check_stack(self, targets):
        targets = targets[0]
        for i in reversed(range(len(targets) - 1)):
            top = targets[i]
            bottom = targets[i + 1]
            pos_top = self.scene_graph_gt[top]['pos']
            pos_bot = self.scene_graph_gt[bottom]['pos']
            if (pos_top[2] - pos_bot[2]) < 0.0:
                return False
            if np.linalg.norm(pos_top[0:2] - pos_bot[0:2]) > 0.1:
                return False
        return True

    def _check_answer(self, answer):
        gt_answer = self.instruction['program'][-1]['output']
        if gt_answer is None:
            return False
        if answer is None:
            return False
        # print(answer)
        if isinstance(answer[0], list):
            if not isinstance(gt_answer[0], list):
                # answer = [a[0] for a in answer]
                answer = answer[0]
                # print(answer)
                if type(answer[0]).__module__ == np.__name__:
                    answer = [a.tolist() for a in answer]
        # print(gt_answer, answer)
        
        if len(answer) != len(gt_answer):
            return False
        for a in answer:
            if a not in gt_answer:
                return False
        return True

    def _make_scene_graph(self, scene, poses, bboxes):
        scene_graph = []
        for i, o in enumerate(scene):
            scene_graph.append(copy.deepcopy(o))
            scene_graph[-1]['pos'] = poses[i][0]
            scene_graph[-1]['ori'] = poses[i][1]
            scene_graph[-1]['bbox'] = bboxes[i]
            scene_graph[-1]['in_hand'] = False
            scene_graph[-1]['raised'] = False
            scene_graph[-1]['approached'] = False
            scene_graph[-1]['gripper_over'] = False
            scene_graph[-1]['weight'] = None
        return scene_graph

    def load_scene_gt(self, scene):
        assert 'objects' in scene
        self.scene_gt = scene
    
    def load_instruction(self, instruction_dict):
        assert 'instruction' in instruction_dict
        self.instruction = instruction_dict

    def _setup_instruction_model(self, params):
        from model.instruction_inference import BaselineSeq2SeqTrainer
        vocab_path = "/media/m2data/NS_AP/NS_AP_v1_0_a/stack/train/vocab.json"
        if 'vocab_path' in params:
            vocab_path = params['vocab_path']
        self.instruction_model = BaselineSeq2SeqTrainer(
            model_params={
                "vocab_path": vocab_path
            },
            loss_params={},
            optimiser_params={},
            scheduler_params={}
        )
        self.instruction_model.set_test()
        self.instruction_model.load_checkpoint(
            params["checkpoint"]
        )

    def _setup_instruction_model_gt(self):
        self.instruction_model_gt = INSTRUCTION_MODELS["GTLoader"]()

    def _setup_visual_recognition_model(self, params):
        from model.visual_recognition import AttributesBaselineTrainer
        self.visual_recognition_model = AttributesBaselineTrainer(
            model_params={
                "ce_num": 5
            },
            loss_params={},
            optimiser_params={},
            scheduler_params={}
        )
        self.visual_recognition_model.set_test()
        self.visual_recognition_model.load_checkpoint(
            params["checkpoint"]
        )

    def _setup_pose_model(self, params):
        assert params['name'] in POSE_MODELS, "Unknown pose model"
        self.pose_model = POSE_MODELS[params['name']](params)

    def _setup_action_planner(self, params):
        from model.action_planner import BaselineActionTrainer
        self.action_planner = BaselineActionTrainer(
            model_params={},
            loss_params={},
            optimiser_params={},
            scheduler_params={}
        )
        self.action_planner.set_test()
        self.action_planner.load_checkpoint(
            params["checkpoint"]
        )

    def _setup_visual_recognition_model_gt(self):
        self.visual_recognition_model_gt = VISUAL_RECOGNITION_MODELS["GTLoader"]()

    def _setup_pose_model_gt(self):
        self.pose_model_gt = POSE_MODELS["GTLoader"]()

    def _setup_segmentation_model(self, params):
        from model.segmentation import BaselineSegmentationTrainer
        self.segmentation_model = BaselineSegmentationTrainer(
            {}, {}, {}, {}
        )
        self.segmentation_model.set_test()
        self.segmentation_model.load_checkpoint(
            params["checkpoint"]
        )

    def _setup_action_planner_gt(self):
        self.action_planner_gt = ACTION_PLAN_MODELS["GTLoader"]()

    def _setup_environment(self):
        assert self.scene_gt is not None, "Load GT scene to initialise simulation"
        controller_cfg_path = self.environment_params.pop('controller_config_path', None)
        if controller_cfg_path is not None:
            controller_cfg = load_controller_config(controller_cfg_path)
        else:
            controller_cfg = None
        self.environment_params['controller_configs'] = controller_cfg
        blender_cfg_path = self.environment_params.pop('blender_config_path', None)
        if blender_cfg_path is not None:
            with open(blender_cfg_path, 'r') as f:
                blender_cfg = json.load(f)
        else:
            blender_cfg = None
        self.environment_params['blender_config'] = blender_cfg
        self.environment_params['scene_dict'] = self.scene_gt
        obj_dict = self.scene_parser.parse_scene(self.scene_gt)
        self.environment_params['objs'] = obj_dict

        self.environment = TabletopEnv(**self.environment_params)
        if 'blender_render' in self.environment_params:
            self.blender_rendering = self.environment_params['blender_render']
        else:
            self.blender_rendering = False

        action_dim = self.environment.action_dim
        self.default_environment_action = np.array(
            (action_dim - 1) * [0] + [-1]
        )

    def _load_image(self, path):
        return Image.open(path)

    def _get_task_from_instruction(self):
        prog_last = self.instruction["program"][-1]
        outp = prog_last["output"]
        if isinstance(prog_last["output"][0], list):
            new_outp = []
            for o_item in prog_last["output"]:
                new_outp += o_item
            outp = [new_outp]
        if 'move' in prog_last["type"]:
            out_task = f"{prog_last['type']}[{prog_last['input_value']}]"
        else:
            out_task = prog_last["type"]
        task = {
            'task': out_task,
            'target': [outp]
        }
        return task

    def _correct_height(self, poses, poses_gt):
        for i in range(len(poses)):
            pos = poses[i][0]
            ori = poses[i][1]
            pos[2] = poses_gt[i][0][2]
            poses[i] = (pos, ori)
        return poses

    def update_sequence_json(self, obs, last_action=None):
        if not self.environment.blender_enabled:
            return
        if self.obs_num == 0:
            json_name = "sequence.json"
            self.json_path = os.path.join(self.save_dir, json_name)
            info_struct = {
                "info": {
                    "image_filename": self.instruction["image_filename"],
                    "instruction": self.instruction["instruction"],
                    "task": self._get_task_from_instruction()
                },
                "observations": [],
                "observations_gt": [],
                "image_paths": [],
                "result": None
            }
            with open(self.json_path, 'w') as f:
                json.dump(info_struct, f, indent=4)

        with open(self.json_path, 'r') as f:
            info_struct = json.load(f)

        info_struct['image_paths'].append(self.last_render_path)
        obs_set = {}
        obs_set['robot'] = {
            'pos': obs['robot0_eef_pos'].tolist(),
            'ori': obs['robot0_eef_quat'].tolist(),
            'gripper_action': obs['gripper_action'].tolist(),
            'gripper_closed': obs['gripper_closed'].tolist(),
            'weight_measurement': obs['weight_measurement'].tolist(),
            'action': last_action
        }

        obs_set['objects'] = []
        for obj in self.scene_graph:
            # print(obj)
            obs_set['objects'].append(copy.deepcopy(obj))
            obs_set['objects'][-1]['bbox'] = obs_set['objects'][-1]['bbox'].tolist()
            obs_set['objects'][-1]['3d_coords'] = obs_set['objects'][-1]['3d_coords'].tolist()
            obs_set['objects'][-1]['orientation'] = obs_set['objects'][-1]['orientation'].tolist()
            obs_set['objects'][-1]['pos'] = obs_set['objects'][-1]['pos'].tolist()
            obs_set['objects'][-1]['ori'] = obs_set['objects'][-1]['ori'].tolist()
            if obs_set['objects'][-1]['weight'] is not None:
                obs_set['objects'][-1]['weight'] = obs_set['objects'][-1]['weight'].tolist()

        obs_set_gt = {}
        robot_body, gripper_body = self.environment.get_robot_configuration()
        for k, v in robot_body.items():
            robot_body[k] = (v[0].tolist(), v[1].tolist())
        for k, v in gripper_body.items():
            gripper_body[k] = (v[0].tolist(), v[1].tolist())

        masks = self.environment.get_segmentation_masks()
        obs_set_gt['robot'] = {
            'pos': obs['robot0_eef_pos'].tolist(),
            'ori': obs['robot0_eef_quat'].tolist(),
            'gripper_action': obs['gripper_action'].tolist(),
            'gripper_closed': obs['gripper_closed'].tolist(),
            'weight_measurement': obs['weight_measurement'].tolist(),
            'grasped_obj_idx': obs['grasped_obj_idx'].tolist(),
            'robot_body': robot_body,
            'gripper_body': gripper_body,
            'robot_mask': masks['robot'],
            'table_mask': masks['table']
        }

        obs_set_gt['objects'] = []
        for i, obj in enumerate(self.scene_graph_gt):
            obs_set_gt['objects'].append(copy.deepcopy(obj))
            obs_set_gt['objects'][i]['mask'] = masks['objects'][i]
            obs_set_gt['objects'][-1]['bbox'] = obs_set_gt['objects'][-1]['bbox'].tolist()
            obs_set_gt['objects'][-1]['pos'] = obs_set_gt['objects'][-1]['pos'].tolist()
            obs_set_gt['objects'][-1]['ori'] = obs_set_gt['objects'][-1]['ori'].tolist()
            if obs_set_gt['objects'][-1]['weight'] is not None:
                obs_set_gt['objects'][-1]['weight'] = obs_set_gt['objects'][-1]['weight'].tolist()


        info_struct['observations'].append(obs_set)
        info_struct['observations_gt'].append(obs_set_gt)
        with open(self.json_path, 'w') as f:
            json.dump(info_struct, f, indent=4)
        self.obs_num += 1

    def outcome_to_json(self, outcome):
        if not self.environment.blender_enabled:
            return

        if self.json_path is None:
            json_name = "sequence.json"
            self.json_path = os.path.join(self.save_dir, json_name)
            info_struct = {
                "info": {
                    "image_filename": self.instruction["image_filename"],
                    "instruction": self.instruction["instruction"],
                    "task": self._get_task_from_instruction()
                },
                "observations": [],
                "observations_gt": [],
                "image_paths": [],
                "result": None
            }
            with open(self.json_path, 'w') as f:
                json.dump(info_struct, f, indent=4)

        with open(self.json_path, 'r') as f:
            info_struct = json.load(f)

        info_struct['result'] = outcome

        with open(self.json_path, 'w') as f:
            json.dump(info_struct, f, indent=4)

    def _detect_loop(self, target_list):
        if len(self.loop_detector) < 2:
            return False
        last_commands = self.loop_detector.get()
        for source_list in last_commands:
            if len(source_list) != len(target_list):
                continue
            else:
                same = True
                for i, command in enumerate(source_list):
                    if command != target_list[i]:
                        same = False
                        break
                if same:
                    return True
        return False

    def _check_gt_scene(self):
        for i in range(len(self.scene_graph_gt)):
            if self.scene_graph_gt[i]['pos'][2] < -0.1:
                return False
        return True

class InferenceToolDebug:
    def __init__(self) -> None:
        self.pose_model = None
        self.instruction_model = None
        self.visual_recognition_model = None
        self.scene_gt = None
        self.instruction = None
        # self.scene = None
        self.blender_rendering = False

        self._last_weight = None

    def setup(self,
              instruction_model_params={},
              visual_recognition_model_params={},
              pose_model_params={},
              action_planner_params={},
              environment_params={},
              program_executor_params={},
              scene_parser_params={},
              action_executor_params={},
              timeout = 10,
              planning_timeout = 20,
              env_timeout = 10000,
              disable_rendering = False, 
              save_dir = 'temp',
              verbose = True,
              move_robot = True
        ):
        self.move_robot = move_robot
        self.verbose = verbose
        self.save_dir = save_dir
        self.disable_rendering = disable_rendering
        self.timeout = timeout
        self.env_timeout = env_timeout
        self.planning_timeout = planning_timeout
        self.environment_params = environment_params
        if self.move_robot:
            self._setup_communication()
        self._setup_instruction_model(instruction_model_params)
        self._setup_visual_recognition_model(visual_recognition_model_params)
        self._setup_pose_model(pose_model_params)
        self._setup_action_planner(action_planner_params)
        self._setup_visual_recognition_model_gt()
        self._setup_pose_model_gt()
        self._setup_action_planner_gt()
        self.scene_parser = SceneParser(**scene_parser_params)
        self.program_executor = ProgramExecutor(**program_executor_params)
        self.action_executor = ActionExecutor(**action_executor_params)
        self.action_executor_robot = ActionExecutor(**action_executor_params)
        self.obs_num = 0
        self.last_render_path = None
        self.loop_detector = CyclicBuffer(2)
        self.prev_pose = None
        self.prev_relative_pose = None
        self.last_grasp_target = None
        self.last_robot_act = None

    def run(self):
        assert self.instruction_model is not None, "Load instruction to program model (or set GT instruction mode) before running inference"
        assert self.visual_recognition_model is not None, "Load visual recognition model"
        assert self.pose_model is not None, "Load pose estimation model (or set GT pose mode) before running inference"


        # Get program from instruction
        program_list = self.instruction_model.get_program(self.instruction)
        if self.verbose:
            print(f'Program:\t{program_list}')

        # Setup environment
        self._setup_environment()

        scene_vis_gt = self.visual_recognition_model_gt.get_scene(None, None, self.scene_gt)

        # First from robot
        if self.move_robot:   
            image_robot, labels_robot, poses_robot, bboxes_robot = self._request_img_pose()
            image_robot.save('./output_shared/test_0.png')
            counter = 1
            assert len(scene_vis_gt) == len(poses_robot), 'Incorrect size'
            poses_robot, bboxes_robot = self._align_robot_debug(scene_vis_gt, labels_robot, poses_robot, bboxes_robot)
            self.environment.apply_external_poses(poses_robot)
        if not self.environment.blender_enabled:
            self.environment.render()
        # print(poses_robot)
        # input()

        # Run some iteration to apply gravity to objects
        action = self.default_environment_action
        _, _, _, _ = self.environment.step(action)
        observation, _, _, _ = self.environment.step(action)
        if self.move_robot:
            observation_robot = self._get_observation_robot(observation)
        # input()
        #DEBUG
        self.action_executor.env = self.environment
        self.action_executor.use_ycb_grasps = True
        # self.action_executor.interpolate_free_movement = True
        self.action_executor_robot.env = self.environment
        self.action_executor_robot.interpolate_free_movement = True
        self.action_executor_robot.decouple = False
        self.action_executor_robot.eps_move_l1_ori = np.pi / 45
        self.action_executor_robot.use_ycb_grasps = True

        self.previous_gripper_action = -1.0
        self.previous_gripper_action_robot = -1.0

        # Get first image
        if self.environment.blender_enabled:
            image_path = self.environment.blender_render()
            self.last_render_path = image_path
            image = self._load_image(image_path)
        else:
            # Without blender we don't use the environment to output images
            # Rather for debugging
            # Can be changed though (save images from MuJoCo)
            if not self.disable_rendering:
                self.environment.render()
            image = None
        # input()

        poses, bboxes = self.pose_model.get_pose(image, observation)
        poses_gt, bboxes_gt = self.pose_model_gt.get_pose(image, observation)
        
        # print(poses, poses_robot)

        # exit()

        scene_vis = self.visual_recognition_model.get_scene(image, None, self.scene_gt)
        

        # print(scene_vis)
        # exit()

        self.scene_graph = self._make_scene_graph(scene_vis, poses, bboxes)
        self.scene_graph_gt = self._make_scene_graph(scene_vis_gt, poses_gt, bboxes_gt)

        if self.move_robot:
            self.scene_graph_robot = self._make_scene_graph(scene_vis, poses_robot, bboxes_robot)
        
        print(f'Move robot: {self.move_robot}')
        # print(self.scene_graph)
        # print(self.scene_graph_robot)
        # exit()

        self.environment.print_robot_configuration()
        # input()

        if self.move_robot:
            observation_robot['action'] = ('START', None)
            observation_robot['action_list'] = [('START', None)]
            self.update_sequence_json(observation, ('START', None), observation_robot)
        elif self.environment.blender_enabled:
            self.update_sequence_json(observation, ('START', None))

        if not self._check_gt_scene():
            print("Broken scene error")
            return InferenceCode.BROKEN_SCENE

        self.task = self._get_task_from_instruction()
        # print(self.scene_graph)
        # exit()
        action_plan_robot = []

        for _ in range(self.timeout):
            # self.scene_graph[0]['weight'] = np.array(160)
            # self.scene_graph[1]['weight'] = np.array(140)
            # self.scene_graph[2]['weight'] = np.array(113.6)
            # self.scene_graph[3]['weight'] = np.array(163.4)
            program_output = self.program_executor.execute(self.scene_graph, program_list)
            self.loop_detector.flush()
            # input()
            print(program_output)
            # print(sce)
            # exit()
            program_status = program_output['STATUS']

            if program_status == ProgramStatus.FAILURE:
                print('Program ended with failure')
                return InferenceCode.PROGRAM_FAILURE

            if program_status == ProgramStatus.SUCCESS:
                print('Program passed through scene graph')
                if self._check_answer(program_output['ANSWER']):
                    print('Answer correct')
                    return InferenceCode.CORRECT_ANSWER
                else:
                    print('Answer incorrect')
                    return InferenceCode.INCORRECT_ANSWER

            if program_status == ProgramStatus.ACTION or program_status == ProgramStatus.FINAL_ACTION:
                planning_tout = self.planning_timeout * len(program_output['ACTION']['target'])
                for _ in range(planning_tout):
                    # print(self.scene_graph[0]['in_hand'])
                    # print(self.scene_graph[0]['gripper_over'])
                    # print(self.scene_graph[0]['approached'])
                    # print(self.scene_graph[0]['raised'])
                    # print(self.scene_graph[1]['in_hand'])
                    # print(self.scene_graph[1]['gripper_over'])
                    # print(self.scene_graph[1]['approached'])
                    # print(self.scene_graph[1]['raised'])
                    # print(observation['robot0_eef_pos'])
                    # input()
                    # print(self.scene_graph_gt[2]['in_hand'], self.scene_graph_gt[2]['raised'])
                    # print(self.scene_graph[2]['in_hand'], self.scene_graph[2]['raised'])
                    # self.scene_graph[2]['in_hand'] = True
                    # self.scene_graph[2]['raised'] = True
                    action_plan = self.action_planner.get_action_sequence(
                        program_output['ACTION'],
                        self.scene_graph
                    )
                    if self.move_robot:
                        action_plan_robot = self.action_planner.get_action_sequence(
                            program_output['ACTION'],
                            self.scene_graph_robot
                        )
                    print(action_plan, action_plan_robot)
                    # if self._detect_loop(action_plan):
                    #     print('Loop detected, exiting')
                    #     return InferenceCode.LOOP_ERROR
                    self.loop_detector.append(action_plan)
                    if len(action_plan) == 0 and len(action_plan_robot) == 0:
                        if program_status == ProgramStatus.ACTION:
                            break
                        if self._check_task_completion(self._get_task_from_instruction(), observation):
                            if program_status == ProgramStatus.FINAL_ACTION:
                                print('Correct execution')
                                return InferenceCode.SUCCESSFUL_TASK
                            else:
                                break
                        else:
                            print('Task not reached target')
                            return InferenceCode.TASK_FAILURE
                    if len(action_plan) > 0:
                        action_to_execute = action_plan[0]
                        self.action_executor.set_action(
                            action_to_execute[0], 
                            action_to_execute[1],
                            observation,
                            self.scene_graph
                        )
                    if len(action_plan_robot) > 0:
                        action_to_execute_robot = action_plan_robot[0]
                        self.action_executor_robot.set_action(
                            action_to_execute_robot[0], 
                            action_to_execute_robot[1],
                            observation_robot,
                            self.scene_graph_robot
                        )
                        self.last_robot_act = action_to_execute_robot[0]
                        if self.last_robot_act == 'approach_grasp':
                            self.last_grasp_target = action_to_execute_robot[1]
                        if self.last_robot_act == 'release':
                            self.last_grasp_target = None
                        
                    action_executed = False
                    # action_executed_robot = False
                    for _ in range(self.env_timeout):
                        # print(self.action_executor.get_current_action(), self.action_executor_robot.get_current_action())
                        if self.move_robot:
                            current_action_present = self.action_executor.get_current_action() or self.action_executor_robot.get_current_action()
                        else:
                            current_action_present = self.action_executor.get_current_action()
                        if current_action_present:
                            if self.action_executor.get_current_action():
                                action = self.action_executor.step(observation)
                            else:
                                action = self.default_environment_action
                                action[6] = self.previous_gripper_action
                            if self.action_executor_robot.get_current_action():
                                action_robot = self.action_executor_robot.step(observation_robot)
                                # print(action_robot)
                                # exit()
                            else:
                                action_robot = self.default_environment_action
                                action_robot[6] = self.previous_gripper_action_robot
                        else:
                            action_executed = True
                            break
                        # print(action, action_robot)
                        # print(action)
                        observation, _, _, _ = self.environment.step(action)
                        if self.move_robot:
                            self._send_robot_action(action_robot, observation_robot)
                        # input()
                        if not self.environment.blender_enabled:
                            if not self.disable_rendering:
                                self.environment.render()
                        if self.move_robot:
                            observation_robot = self._get_observation_robot(observation)
                        self.previous_gripper_action = action[6]
                        self.previous_gripper_action_robot = action_robot[6]
                        # print(observation['gripper_action'])
                        # print(observation['robot0_eef_quat'], observation_robot['robot0_eef_quat'])
                        
                    if action_executed:
                        if self.environment.blender_enabled:
                            image_path = self.environment.blender_render()
                            self.last_render_path = image_path
                            image = self._load_image(image_path)
                        poses, bboxes = self.pose_model.get_pose(image, observation)
                        poses_gt, bboxes_gt = self.pose_model_gt.get_pose(image, observation)
                        if self.move_robot:
                            image_robot, labels_robot, poses_robot, bboxes_robot = self._request_img_pose()
                            poses_robot, bboxes_robot = self._align_robot_debug(
                                scene_vis, labels_robot, poses_robot, bboxes_robot, observation_robot
                            )
                            image_robot.save(f'./output_shared/test_{counter}.png')
                            counter += 1
                        
                        self._update_scene_graph(poses, bboxes, observation)
                        if self.move_robot:
                            self._update_scene_graph(poses_robot, bboxes_robot, observation_robot, robot=True)
                        self._update_scene_graph(poses_gt, bboxes_gt, observation, gt=True)
                        if not self._check_gt_scene():
                            print("Broken scene error")
                            return InferenceCode.BROKEN_SCENE
                        if self.move_robot:
                            observation_robot['action'] = action_to_execute_robot
                            observation_robot['action_list'] = action_plan_robot
                            self.update_sequence_json(observation, action_to_execute, observation_robot)
                        elif self.environment.blender_enabled:
                            self.update_sequence_json(observation, action_to_execute)
                    else:
                        print('Action not executed correctly')
                        return InferenceCode.EXECUTION_ERROR


        print("Timeout, program execution reiterations exceeded.")
        return InferenceCode.TIMEOUT

    # def _load_scene(self, scene):
    #     assert 'objects' in scene
    #     self.scene = scene

    def _update_scene_graph(self, poses, bboxes, obs, gt=False, robot=False):
        if gt:
            for i in range(len(self.scene_graph_gt)):
                self.scene_graph_gt[i]['pos'] = poses[i][0]
                self.scene_graph_gt[i]['ori'] = poses[i][1]
                self.scene_graph_gt[i]['bbox'] = bboxes[i]
            eef_pos = obs['robot0_eef_pos']
            eef_ori = obs['robot0_eef_quat']
            gripper_closed = obs['gripper_closed']
            gripper_action = obs['gripper_action']
            # Iterate again cause ultimately there will be matching 
            for idx, obj in enumerate(self.scene_graph_gt):
                obj['in_hand'] = False
                obj['raised'] = False
                obj['approached'] = False
                obj['gripper_over'] = False
                bbox_boundaries = self._check_eef_in_bbox(obj, eef_pos)
                finger_in_bbox = self._check_finger_in_bbox(obj, eef_pos, eef_ori)
                # print(obj['name'], finger_in_bbox)
                # print(bbox_boundaries)
                # print(gripper_closed)
                # print(gripper_action)
                if len(bbox_boundaries) == 3 or finger_in_bbox:
                    if gripper_closed and gripper_action > -0.99:
                        obj['in_hand'] = True
                    else:
                        obj['approached'] = True
                # elif ('x' in bbox_boundaries and 'y' in bbox_boundaries):
                elif len(bbox_boundaries) == 2:
                    obj_mat = T.quat2mat(obj['ori'])
                    obj_z_dir = obj_mat[:, 2]
                    obj_y_dir = obj_mat[:, 1]
                    obj_x_dir = obj_mat[:, 0]
                    angle_to_z = self._angle_between(obj_z_dir, np.array([0, 0, 1]))
                    angle_to_y = self._angle_between(obj_y_dir, np.array([0, 0, 1]))
                    angle_to_x = self._angle_between(obj_x_dir, np.array([0, 0, 1]))
                    if np.abs(angle_to_z) < np.pi / 6 or np.abs(angle_to_z) - np.pi < np.pi / 6:
                        if ('x' in bbox_boundaries and 'y' in bbox_boundaries):
                            obj['gripper_over'] = True
                    elif np.abs(angle_to_y) < np.pi / 6 or np.abs(angle_to_y) - np.pi < np.pi / 6:
                        if ('x' in bbox_boundaries and 'z' in bbox_boundaries):
                            obj['gripper_over'] = True
                    elif np.abs(angle_to_x) < np.pi / 6 or np.abs(angle_to_x) - np.pi < np.pi / 6:
                        if ('z' in bbox_boundaries and 'y' in bbox_boundaries):
                            obj['gripper_over'] = True    
                if all([obj['bbox'][i][2] > 0.035 for i in range(8)]):
                    obj['raised'] = True
                    if obs['grasped_obj_idx'] == idx:
                        obj['weight'] = obs['weight_measurement']
        elif robot:
            #TODO
            # Do the pose matching w.r.t history
            for i in range(len(self.scene_graph_robot)):
                self.scene_graph_robot[i]['pos'] = poses[i][0]
                self.scene_graph_robot[i]['ori'] = poses[i][1]
                self.scene_graph_robot[i]['bbox'] = bboxes[i]
            eef_pos = obs['robot0_eef_pos']
            eef_ori = obs['robot0_eef_quat']
            gripper_closed = obs['gripper_closed']
            gripper_action = obs['gripper_action']
            # print(obs)
            # Iterate again cause ultimately there will be matching 
            for obj in self.scene_graph_robot:
                obj['in_hand'] = False
                obj['raised'] = False
                obj['approached'] = False
                obj['gripper_over'] = False
                bbox_boundaries = self._check_eef_in_bbox(obj, eef_pos)
                finger_in_bbox = self._check_finger_in_bbox(obj, eef_pos, eef_ori)
                # print(bbox_boundaries)
                # print(gripper_closed)
                # print(gripper_action)
                if len(bbox_boundaries) == 3 or finger_in_bbox:
                    if gripper_closed and gripper_action > -0.99:
                        obj['in_hand'] = True
                    else:
                        obj['approached'] = True
                # elif ('x' in bbox_boundaries and 'y' in bbox_boundaries):
                elif len(bbox_boundaries) == 2:
                    obj_mat = T.quat2mat(obj['ori'])
                    obj_z_dir = obj_mat[:, 2]
                    obj_y_dir = obj_mat[:, 1]
                    obj_x_dir = obj_mat[:, 0]
                    angle_to_z = self._angle_between(obj_z_dir, np.array([0, 0, 1]))
                    angle_to_y = self._angle_between(obj_y_dir, np.array([0, 0, 1]))
                    angle_to_x = self._angle_between(obj_x_dir, np.array([0, 0, 1]))
                    if np.abs(angle_to_z) < np.pi / 6 or np.abs(angle_to_z) - np.pi < np.pi / 6:
                        if ('x' in bbox_boundaries and 'y' in bbox_boundaries):
                            obj['gripper_over'] = True
                    elif np.abs(angle_to_y) < np.pi / 6 or np.abs(angle_to_y) - np.pi < np.pi / 6:
                        if ('x' in bbox_boundaries and 'z' in bbox_boundaries):
                            obj['gripper_over'] = True
                    elif np.abs(angle_to_x) < np.pi / 6 or np.abs(angle_to_x) - np.pi < np.pi / 6:
                        if ('z' in bbox_boundaries and 'y' in bbox_boundaries):
                            obj['gripper_over'] = True    
                if all([obj['bbox'][i][2] > 0.035 for i in range(8)]):
                    obj['raised'] = True
                if obj['raised'] and obj['in_hand']:
                    obj['weight'] = obs['weight_measurement']
        else:
            #TODO
            # Do the pose matching w.r.t history
            for i in range(len(self.scene_graph)):
                self.scene_graph[i]['pos'] = poses[i][0]
                self.scene_graph[i]['ori'] = poses[i][1]
                self.scene_graph[i]['bbox'] = bboxes[i]
            eef_pos = obs['robot0_eef_pos']
            eef_ori = obs['robot0_eef_quat']
            gripper_closed = obs['gripper_closed']
            gripper_action = obs['gripper_action']
            # print(obs)
            # Iterate again cause ultimately there will be matching 
            for obj in self.scene_graph:
                obj['in_hand'] = False
                obj['raised'] = False
                obj['approached'] = False
                obj['gripper_over'] = False
                bbox_boundaries = self._check_eef_in_bbox(obj, eef_pos)
                finger_in_bbox = self._check_finger_in_bbox(obj, eef_pos, eef_ori)
                # print(bbox_boundaries)
                # print(gripper_closed)
                # print(gripper_action)
                if len(bbox_boundaries) == 3 or finger_in_bbox:
                    if gripper_closed and gripper_action > -0.99:
                        obj['in_hand'] = True
                    else:
                        obj['approached'] = True
                # elif ('x' in bbox_boundaries and 'y' in bbox_boundaries):
                elif len(bbox_boundaries) == 2:
                    obj_mat = T.quat2mat(obj['ori'])
                    obj_z_dir = obj_mat[:, 2]
                    obj_y_dir = obj_mat[:, 1]
                    obj_x_dir = obj_mat[:, 0]
                    angle_to_z = self._angle_between(obj_z_dir, np.array([0, 0, 1]))
                    angle_to_y = self._angle_between(obj_y_dir, np.array([0, 0, 1]))
                    angle_to_x = self._angle_between(obj_x_dir, np.array([0, 0, 1]))
                    if np.abs(angle_to_z) < np.pi / 6 or np.abs(angle_to_z) - np.pi < np.pi / 6:
                        if ('x' in bbox_boundaries and 'y' in bbox_boundaries):
                            obj['gripper_over'] = True
                    elif np.abs(angle_to_y) < np.pi / 6 or np.abs(angle_to_y) - np.pi < np.pi / 6:
                        if ('x' in bbox_boundaries and 'z' in bbox_boundaries):
                            obj['gripper_over'] = True
                    elif np.abs(angle_to_x) < np.pi / 6 or np.abs(angle_to_x) - np.pi < np.pi / 6:
                        if ('z' in bbox_boundaries and 'y' in bbox_boundaries):
                            obj['gripper_over'] = True           
                if all([obj['bbox'][i][2] > 0.035 for i in range(8)]):
                    obj['raised'] = True
                if obj['raised'] and obj['in_hand']:
                    obj['weight'] = obs['weight_measurement']
                # print(obj['name'], obj['in_hand'], obj['raised'])

    def _angle_between(self, v1, v2):
        v1_u = self._unit_vector(v1)
        v2_u = self._unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    def _unit_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def _check_eef_in_bbox(self, obj, eef_pos):
        bbox = obj['bbox']
        x_vec = bbox[0] - bbox[1]
        y_vec = bbox[0] - bbox[3]
        z_vec = bbox[4] - bbox[0]
        x_size = np.linalg.norm(x_vec)
        y_size = np.linalg.norm(y_vec)
        z_size = np.linalg.norm(z_vec)
        # print(bbox.shape)
        bbox_mid = np.mean(bbox, axis=0)
        mid_to_eef_vec = eef_pos - bbox_mid
        axes_in = []
        x_proj = np.abs(np.dot(mid_to_eef_vec, x_vec) / x_size)
        if x_proj < x_size / 2:
            axes_in.append('x')
        y_proj = np.abs(np.dot(mid_to_eef_vec, y_vec) / y_size)
        if y_proj < y_size / 2:
            axes_in.append('y')
        z_proj = np.abs(np.dot(mid_to_eef_vec, z_vec) / z_size)
        if z_proj < z_size / 2:
            axes_in.append('z')
        return axes_in 

    def _check_finger_in_bbox(self, obj, eef_pos, eef_ori):
        finger_dist = self.action_executor.gripper_open_width
        gripper_y_dir = T.quat2mat(eef_ori)[:, 1]
        gripper_y_dir = gripper_y_dir / np.linalg.norm(gripper_y_dir)
        finger1 = eef_pos + finger_dist * gripper_y_dir
        finger2 = eef_pos - finger_dist * gripper_y_dir
        if len(self._check_eef_in_bbox(obj, finger1)) == 3:
            return True
        if len(self._check_eef_in_bbox(obj, finger2)) == 3:
            return True
        return False


    def _check_task_completion(self, action, obs):
        if action['task'] == 'measure_weight':
            if self.scene_graph_gt[action['target'][0][0]]['raised']:
                if obs['weight_measurement'] > 0:
                    self.scene_graph_gt[action['target'][0][0]]['weight'] = obs['weight_measurement']
                    return True
        elif action['task'] == 'pick_up':
            print(action['target'][0][0])
            if self.scene_graph_gt[action['target'][0][0]]['raised']:
                 return True
        elif action['task'] == 'stack':
            return self._check_stack(action['target'][0])
        elif 'move' in action['task']:
            return self._check_move(action)
        return False

    def _check_move(self, task):
        side = task['task'].split('[')[1].strip(']')
        for idx in task['target'][0]:
            if self.scene_graph_gt[idx]['in_hand']:
                return False
            y_pos = self.scene_graph_gt[idx]['pos'][1]
            if (side == 'right' and y_pos < 0):
                return False
            if (side == 'left' and y_pos > 0):
                return False
        return True

    def _check_stack(self, targets):
        targets = targets[0]
        for i in reversed(range(len(targets) - 1)):
            top = targets[i]
            bottom = targets[i + 1]
            pos_top = self.scene_graph_gt[top]['pos']
            pos_bot = self.scene_graph_gt[bottom]['pos']
            if (pos_top[2] - pos_bot[2]) < 0.0:
                return False
            if np.linalg.norm(pos_top[0:2] - pos_bot[0:2]) > 0.1:
                return False
        return True

    def _check_answer(self, answer):
        gt_answer = self.instruction['program'][-1]['output']
        if gt_answer is None:
            return False
        if answer is None:
            return False
        # print(answer)
        if isinstance(answer[0], list):
            if not isinstance(gt_answer[0], list):
                # answer = [a[0] for a in answer]
                answer = answer[0]
                # print(answer)
                if type(answer[0]).__module__ == np.__name__:
                    answer = [a.tolist() for a in answer]
        # print(gt_answer, answer)
        
        if len(answer) != len(gt_answer):
            return False
        for a in answer:
            if a not in gt_answer:
                return False
        return True

    def _make_scene_graph(self, scene, poses, bboxes):
        scene_graph = []
        for i, o in enumerate(scene):
            scene_graph.append(copy.deepcopy(o))
            scene_graph[-1]['pos'] = poses[i][0]
            scene_graph[-1]['ori'] = poses[i][1]
            scene_graph[-1]['bbox'] = bboxes[i]
            scene_graph[-1]['in_hand'] = False
            scene_graph[-1]['raised'] = False
            scene_graph[-1]['approached'] = False
            scene_graph[-1]['gripper_over'] = False
        return scene_graph

    def load_scene_gt(self, scene):
        assert 'objects' in scene
        self.scene_gt = scene
    
    def load_instruction(self, instruction_dict):
        assert 'instruction' in instruction_dict
        self.instruction = instruction_dict

    def _setup_instruction_model(self, params):
        assert params['name'] in INSTRUCTION_MODELS, "Unknown instruction model"
        self.instruction_model = INSTRUCTION_MODELS[params['name']]()

    def _setup_visual_recognition_model(self, params):
        assert params['name'] in VISUAL_RECOGNITION_MODELS, "Unknown visual model"
        self.visual_recognition_model = VISUAL_RECOGNITION_MODELS[params['name']]()

    def _setup_pose_model(self, params):
        assert params['name'] in POSE_MODELS, "Unknown pose model"
        self.pose_model = POSE_MODELS[params['name']]()

    def _setup_action_planner(self, params):
        assert params['name'] in ACTION_PLAN_MODELS, "Unknown action planner"
        self.action_planner = ACTION_PLAN_MODELS[params['name']]()

    def _setup_visual_recognition_model_gt(self):
        self.visual_recognition_model_gt = VISUAL_RECOGNITION_MODELS["GTLoader"]()

    def _setup_pose_model_gt(self):
        self.pose_model_gt = POSE_MODELS["GTLoader"]()

    def _setup_action_planner_gt(self):
        self.action_planner_gt = ACTION_PLAN_MODELS["GTLoader"]()

    def _setup_environment(self):
        assert self.scene_gt is not None, "Load GT scene to initialise simulation"
        controller_cfg_path = self.environment_params.pop('controller_config_path', None)
        if controller_cfg_path is not None:
            controller_cfg = load_controller_config(controller_cfg_path)
        else:
            controller_cfg = None
        self.environment_params['controller_configs'] = controller_cfg
        blender_cfg_path = self.environment_params.pop('blender_config_path', None)
        if blender_cfg_path is not None:
            with open(blender_cfg_path, 'r') as f:
                blender_cfg = json.load(f)
        else:
            blender_cfg = None
        self.environment_params['blender_config'] = blender_cfg
        self.environment_params['scene_dict'] = self.scene_gt
        obj_dict = self.scene_parser.parse_scene(self.scene_gt)
        self.environment_params['objs'] = obj_dict

        self.environment = TabletopEnv(**self.environment_params)
        if 'blender_render' in self.environment_params:
            self.blender_rendering = self.environment_params['blender_render']
        else:
            self.blender_rendering = False

        action_dim = self.environment.action_dim
        self.default_environment_action = np.array(
            (action_dim - 1) * [0] + [-1]
        )

    def _load_image(self, path):
        return Image.open(path)

    def _get_task_from_instruction(self):
        prog_last = self.instruction["program"][-1]
        outp = prog_last["output"]
        if isinstance(prog_last["output"][0], list):
            new_outp = []
            for o_item in prog_last["output"]:
                new_outp += o_item
            outp = [new_outp]
        if 'move' in prog_last["type"]:
            out_task = f"{prog_last['type']}[{prog_last['input_value']}]"
        else:
            out_task = prog_last["type"]
        task = {
            'task': out_task,
            'target': [outp]
        }
        return task

    def update_sequence_json(self, obs, last_action=None, obs_robot=None):
        if not self.environment.blender_enabled:
            return
        if self.obs_num == 0:
            json_name = "sequence.json"
            self.json_path = os.path.join(self.save_dir, json_name)
            info_struct = {
                "info": {
                    "image_filename": self.instruction["image_filename"],
                    "instruction": self.instruction["instruction"],
                    "task": self._get_task_from_instruction()
                },
                "observations": [],
                "observations_gt": [],
                "image_paths": [],
                "result": None
            }
            if obs_robot is not None:
                info_struct['observations_robot'] = []
            with open(self.json_path, 'w') as f:
                json.dump(info_struct, f, indent=4)

        with open(self.json_path, 'r') as f:
            info_struct = json.load(f)

        info_struct['image_paths'].append(self.last_render_path)
        obs_set = {}
        obs_set['robot'] = {
            'pos': obs['robot0_eef_pos'].tolist(),
            'ori': obs['robot0_eef_quat'].tolist(),
            'gripper_action': obs['gripper_action'].tolist(),
            'gripper_closed': obs['gripper_closed'].tolist(),
            'weight_measurement': obs['weight_measurement'].tolist(),
            'action': last_action
        }

        obs_set['objects'] = []
        for obj in self.scene_graph:
            obs_set['objects'].append(copy.deepcopy(obj))
            obs_set['objects'][-1]['bbox'] = obs_set['objects'][-1]['bbox'].tolist()
            obs_set['objects'][-1]['pos'] = obs_set['objects'][-1]['pos'].tolist()
            obs_set['objects'][-1]['ori'] = obs_set['objects'][-1]['ori'].tolist()
            if obs_set['objects'][-1]['weight'] is not None:
                obs_set['objects'][-1]['weight'] = obs_set['objects'][-1]['weight'].tolist()

        if obs_robot is not None:
            obs_set_robot = {}
            obs_set_robot['robot'] = {
                'pos': obs_robot['robot0_eef_pos'],
                'ori': obs_robot['robot0_eef_quat'],
                'gripper_action': obs_robot['gripper_action'],
                'gripper_closed': obs_robot['gripper_closed'],
                'weight_measurement': obs_robot['weight_measurement']
            }
            if 'action' in obs_robot:
                obs_set_robot['robot']['action'] = obs_robot['action']
            if 'action_list' in obs_robot:
                obs_set_robot['robot']['action_list'] = obs_robot['action_list']

            obs_set_robot['objects'] = []
            for obj in self.scene_graph_robot:
                obs_set_robot['objects'].append(copy.deepcopy(obj))
                obs_set_robot['objects'][-1]['bbox'] = obs_set_robot['objects'][-1]['bbox'].tolist()
                obs_set_robot['objects'][-1]['pos'] = obs_set_robot['objects'][-1]['pos'].tolist()
                obs_set_robot['objects'][-1]['ori'] = obs_set_robot['objects'][-1]['ori'].tolist()
                if obs_set_robot['objects'][-1]['weight'] is not None:
                    obs_set_robot['objects'][-1]['weight'] = obs_set_robot['objects'][-1]['weight']


        obs_set_gt = {}
        robot_body, gripper_body = self.environment.get_robot_configuration()
        for k, v in robot_body.items():
            robot_body[k] = (v[0].tolist(), v[1].tolist())
        for k, v in gripper_body.items():
            gripper_body[k] = (v[0].tolist(), v[1].tolist())

        masks = self.environment.get_segmentation_masks()
        obs_set_gt['robot'] = {
            'pos': obs['robot0_eef_pos'].tolist(),
            'ori': obs['robot0_eef_quat'].tolist(),
            'gripper_action': obs['gripper_action'].tolist(),
            'gripper_closed': obs['gripper_closed'].tolist(),
            'weight_measurement': obs['weight_measurement'].tolist(),
            'grasped_obj_idx': obs['grasped_obj_idx'].tolist(),
            'robot_body': robot_body,
            'gripper_body': gripper_body,
            'robot_mask': masks['robot'],
            'table_mask': masks['table']
        }

        obs_set_gt['objects'] = []
        for i, obj in enumerate(self.scene_graph_gt):
            obs_set_gt['objects'].append(copy.deepcopy(obj))
            obs_set_gt['objects'][i]['mask'] = masks['objects'][i]
            obs_set_gt['objects'][-1]['bbox'] = obs_set_gt['objects'][-1]['bbox'].tolist()
            obs_set_gt['objects'][-1]['pos'] = obs_set_gt['objects'][-1]['pos'].tolist()
            obs_set_gt['objects'][-1]['ori'] = obs_set_gt['objects'][-1]['ori'].tolist()
            if obs_set_gt['objects'][-1]['weight'] is not None:
                obs_set_gt['objects'][-1]['weight'] = obs_set_gt['objects'][-1]['weight'].tolist()


        info_struct['observations'].append(obs_set)
        info_struct['observations_gt'].append(obs_set_gt)
        if self.move_robot:
            info_struct['observations_robot'].append(obs_set_robot)
        with open(self.json_path, 'w') as f:
            json.dump(info_struct, f, indent=4)
        self.obs_num += 1

    def outcome_to_json(self, outcome):
        if not self.environment.blender_enabled:
            return
        with open(self.json_path, 'r') as f:
            info_struct = json.load(f)

        info_struct['result'] = outcome

        with open(self.json_path, 'w') as f:
            json.dump(info_struct, f, indent=4)

    def _detect_loop(self, target_list):
        if len(self.loop_detector) < 2:
            return False
        last_commands = self.loop_detector.get()
        for source_list in last_commands:
            if len(source_list) != len(target_list):
                continue
            else:
                same = True
                for i, command in enumerate(source_list):
                    if command != target_list[i]:
                        same = False
                        break
                if same:
                    return True
        return False

    def _check_gt_scene(self):
        for i in range(len(self.scene_graph_gt)):
            if self.scene_graph_gt[i]['pos'][2] < -0.1:
                return False
        return True

    def _get_observation_robot(self, observation):
        observation_robot = observation.copy()
        pos, ori, gripper_action, gripper_closed, weight = self._pool_gripper_data(observation)
        observation_robot['robot0_eef_pos'] = pos
        observation_robot['robot0_eef_quat'] = ori
        observation_robot['gripper_action'] = gripper_action
        observation_robot['gripper_closed'] = gripper_closed
        observation_robot['weight_measurement'] = weight

        return observation_robot

    def _pool_gripper_data(self, obs):
        msg = self._socket_state_msg.recv()
        state = pickle.loads(msg)
        pos = np.array(state['eef_trans'])
        ori = np.array(state['eef_rot'])
        joint_state = state['joint_states']
        finger1_pos = joint_state['position'][-2]
        finger2_pos = joint_state['position'][-1]
        finger1_vel = joint_state['velocity'][-2]
        finger2_vel = joint_state['velocity'][-1]
        finger_reach = 0.03988
        if finger1_pos < 0.039 and finger2_pos < 0.039 and np.abs(finger1_vel) < 0.001 and np.abs(finger2_vel) < 0.001:
            gripper_closed = True
        else:
            gripper_closed = False
        
        finger1_perc_close = 1.0 - finger1_pos / finger_reach
        finger2_perc_close = 1.0 - finger2_pos / finger_reach
        gripper_close_perc = 0.5 * (finger1_perc_close + finger2_perc_close)
        gripper_action = 2 * gripper_close_perc - 1.0
        # print(f'gripper_action: {gripper_action}')
        weight_msg = self._last_weight
        # weight_msg = self._weight_client.wait_receive_param("F_ext", timeout=1000)
        z_force = weight_msg['wrench']['force']['z']
        weight = -z_force / 9.81


        # if finger1_vel > 0.001 and finger2_vel > 0.001:
        #     gripper_action = -1.0
        # else:
        #     gripper_action = 1.0

        # joint_state = 
        # print(state)
        # print(pos)
        # print(ori)

        # pos = obs['robot0_eef_pos'] 
        # ori = obs['robot0_eef_quat'] 
        # print(pos)
        # print(ori)
        # print(gripper_action, gripper_closed)
        # gripper_action = obs['gripper_action'] 
        # gripper_closed = obs['gripper_closed'] 
        # weight = obs['weight_measurement'] 

        return pos, ori, gripper_action, gripper_closed, weight

    def _send_robot_action(self, action, obs):
        pos = obs['robot0_eef_pos']
        ori = obs['robot0_eef_quat']
        print(pos, ori)
        des_pos = pos + action[0:3]
        # print(pos, des_pos)
        des_ori = T.quat_multiply(ori, T.axisangle2quat(action[3:6]))
        # print(des_pos)
        if all([des_ori[i] < 0 for i in range(3)]):
            des_ori = - des_ori
        shift_to_wrist = np.matmul(T.quat2mat(des_ori), np.array([0, 0, -0.103]))
        des_pos += shift_to_wrist 
        # print(des_pos)
        if action[6] > 0:
            gripper_action = b'close'
        else:
            gripper_action = b'open'
    
        des_pose = des_pos.tolist() + des_ori.tolist()
        des_pose = map(lambda x: '%.6f' % x, des_pose)
        msg = ' '.join(des_pose)
        print(msg)
        # exit()
        self._socket_pose_control.send_string(msg)
        if gripper_action != self.gripper_msg_prev:
            self._socket_gripper_control.send(gripper_action)
            self._socket_gripper_control.recv()
            self.gripper_msg_prev = gripper_action

        # time.sleep(2.0)

    def _request_img_pose(self):
        self._socket_img_pose_msg.send_string("scene_camera")
        # self._socket_img_pose_msg.send(b'0')
        print('msg send')
        msg = self._socket_img_pose_msg.recv()
        print('msg received')
        msg = pickle.loads(msg)
        img = msg['image']
        img = Image.fromarray(img)
        objs = msg['objects']
        names, poses, bboxes = [], [], []
        for obj in objs:
            name = COSYPOSE2NAME[obj['label']]
            names.append(name)
            print(name)
            pose_cosypose = obj['pose']
            pos_cosy, ori_cosy = pose_cosypose
            pos_correction = np.matmul(
                T.quat2mat(ori_cosy), COSYPOSE_TRANSFORM[obj['label']][0]
            )
            pos = pos_cosy + pos_correction
            ori = T.quat_multiply(ori_cosy, COSYPOSE_TRANSFORM[obj['label']][1])
            print(pos_cosy, pos)
            print(ori_cosy, ori)

            bbox_xyz = COSYPOSE_BBOX[obj['label']]
            bbox_local = self._get_local_bounding_box(bbox_xyz)
            bbox_local = np.concatenate(
                (
                    bbox_local.T,
                    np.ones((bbox_local.shape[0], 1)).T
                )
            )
            pose_mat = T.pose2mat((pos, ori))
            # print(pose_mat)
            bbox_world = np.matmul(pose_mat, bbox_local)
            poses.append((pos, ori))
            bboxes.append(bbox_world[:-1, :].T)
        # exit()
        
        return img, names, poses, bboxes

    def _setup_communication(self):
        context = zmq.Context()
        self._socket_state_msg = context.socket(zmq.SUB)
        self._socket_state_msg.setsockopt(zmq.SUBSCRIBE, b'')
        self._socket_state_msg.setsockopt(zmq.CONFLATE, True)
        self._socket_state_msg.connect("tcp://127.0.0.1:5555")
        self._socket_gripper_control = context.socket(zmq.REQ)
        self._socket_gripper_control.connect("tcp://127.0.0.1:5556")
        self._socket_pose_control = context.socket(zmq.PUB)
        self._socket_pose_control.bind("tcp://127.0.0.1:5557")
        self._socket_img_pose_msg = context.socket(zmq.REQ)
        self._socket_img_pose_msg.connect("tcp://127.0.0.1:5559")
        
        def update_params(parameter, data):
            if parameter != "F_ext":
                return
            if data is None:
                return
            self._last_weight = data

        self._weight_client = ParamSubscriber(
            addr='127.0.0.1',
            start_port=5560
        )
        self._weight_client.declare('F_ext')
        self._weight_client.subscribe('F_ext')
        self._weight_client.set_callback(update_params)

        self.gripper_msg_prev = None

        # self._socket_gripper_control.send(b'close')
        # self._socket_gripper_control.recv()
        # self._socket_gripper_control.send(b'open')
        # exit()

    def _get_local_bounding_box(self, bbox):
        bbox_wh = bbox['x'] / 2
        bbox_dh = bbox['y'] / 2
        bbox_h = bbox['z']

        # Local bounding box w.r.t to local (0,0,0) - mid bottom
        return np.array(
            [
                [ bbox_wh,  bbox_dh, 0],
                [-bbox_wh,  bbox_dh, 0],
                [-bbox_wh, -bbox_dh, 0],
                [ bbox_wh, -bbox_dh, 0],
                [ bbox_wh,  bbox_dh, bbox_h],
                [-bbox_wh,  bbox_dh, bbox_h],
                [-bbox_wh, -bbox_dh, bbox_h],
                [ bbox_wh, -bbox_dh, bbox_h]
            ]
        )

    def _align_robot_debug(self, scene, labels, poses, bboxes, obs=None):
        obj_list = [o['name'] for o in scene]
        if obs is None:
            p_aligned, bb_aligned = [], []
            self.prev_pose = {}
            self.prev_relative_pose = {}
            for o in scene:
                name = o['name']
                idx = labels.index(name)
                p_aligned.append(poses[idx])
                bb_aligned.append(bboxes[idx])
                self.prev_pose[name] = p_aligned
                self.prev_relative_pose[name] = p_aligned
        else:
            reverse_bbox_dict = {
                v: k for k, v in COSYPOSE2NAME.items()
            }
            pos_eef = np.array(obs['robot0_eef_pos'])
            ori_eef = np.array(obs['robot0_eef_quat'])
            world_in_eef = T.quat_inverse(ori_eef)
            p_aligned, bb_aligned = [None] * len(obj_list), [None] * len(obj_list)
            missing = [name for name in obj_list]
            # Fill correct
            for i, name in enumerate(labels):
                if name in obj_list:
                    missing.remove(name)
                    idx = obj_list.index(name)
                    p_aligned[idx] = poses[i]
                    bb_aligned[idx] = bboxes[i]
                    self.prev_pose[name] = poses[i]
                    relative_pos = poses[i][0] - pos_eef
                    # world_in_eef * obj_in_world
                    relative_ori = T.quat_multiply(world_in_eef, poses[i][1])
                    self.prev_relative_pose[name] = (relative_pos, relative_ori)
            if len(missing) > 0:
                print('Objects missing, filling with previous values')
                print(missing)
                gripper_action = obs['gripper_action']
                gripper_on_obj = -0.98 < gripper_action < 0.98
                # If gripper on any object:
                if gripper_on_obj:
                    for name in missing:
                        idx_missing = obj_list.index(name)
                        if self.last_grasp_target is not None and self.last_grasp_target == idx_missing:
                            # If we had previously approach grasp, get its target last known relative pose
                            obj_rel_pose = self.prev_relative_pose[name]
                            obj_pos = pos_eef + obj_rel_pose[0]
                            obj_ori = T.quat_multiply(ori_eef, obj_rel_pose[1])
                            self.prev_pose[name] = (obj_pos, obj_ori)
                            p_aligned[idx_missing] = self.prev_pose[name]
                        else:
                            # If it's any other object, keep previous pose
                            obj_pose = self.prev_pose[name]
                            relative_pos = obj_pose[0] - pos_eef
                            # world_in_eef * obj_in_world
                            relative_ori = T.quat_multiply(world_in_eef, obj_pose[1])
                            self.prev_relative_pose[name] = (relative_pos, relative_ori)
                            p_aligned[idx_missing] = obj_pose
                        bbox_xyz = COSYPOSE_BBOX[reverse_bbox_dict[name]]
                        bbox_local = self._get_local_bounding_box(bbox_xyz)
                        bbox_local = np.concatenate(
                            (
                                bbox_local.T,
                                np.ones((bbox_local.shape[0], 1)).T
                            )
                        )
                        pose_mat = T.pose2mat(p_aligned[idx_missing])
                        # print(pose_mat)
                        bbox_world = np.matmul(pose_mat, bbox_local)[:-1, :].T
                        bb_aligned[idx_missing] = bbox_world
                else:
                    # If gripper not on any object we copy the previous pose
                    # The only tricky case is lack of detection after approach_grasp
                    # However, up to assumption of not having moved the object such
                    # that it is still not detected after moving we should be fine
                    for name in missing:
                        idx_missing = obj_list.index(name)
                        obj_pose = self.prev_pose[name]
                        relative_pos = obj_pose[0] - pos_eef
                        # world_in_eef * obj_in_world
                        relative_ori = T.quat_multiply(world_in_eef, obj_pose[1])
                        self.prev_relative_pose[name] = (relative_pos, relative_ori)
                        p_aligned[idx_missing] = obj_pose
                        bbox_xyz = COSYPOSE_BBOX[reverse_bbox_dict[name]]
                        bbox_local = self._get_local_bounding_box(bbox_xyz)
                        bbox_local = np.concatenate(
                            (
                                bbox_local.T,
                                np.ones((bbox_local.shape[0], 1)).T
                            )
                        )
                        pose_mat = T.pose2mat(p_aligned[idx_missing])
                        # print(pose_mat)
                        bbox_world = np.matmul(pose_mat, bbox_local)[:-1, :].T
                        bb_aligned[idx_missing] = bbox_world
        print('Poses, bboxes')
        print(p_aligned)
        print(bb_aligned)
        return p_aligned, bb_aligned


class InferenceToolGraspTest:
    def __init__(self) -> None:
        self.pose_model = None
        self.instruction_model = None
        self.visual_recognition_model = None
        self.scene_gt = None
        self.instruction = None
        # self.scene = None
        self.blender_rendering = False

    def setup(self,
              instruction_model_params={},
              visual_recognition_model_params={},
              pose_model_params={},
              action_planner_params={},
              environment_params={},
              program_executor_params={},
              scene_parser_params={},
              action_executor_params={},
              timeout = 10,
              planning_timeout = 20,
              env_timeout = 10000,
              disable_rendering = False, 
              save_dir = 'temp',
              verbose = True
        ):
        self.move_robot = True
        self.verbose = verbose
        self.save_dir = save_dir
        self.disable_rendering = disable_rendering
        self.timeout = timeout
        self.env_timeout = env_timeout
        self.planning_timeout = planning_timeout
        self.environment_params = environment_params
        self._setup_communication()
        self._setup_instruction_model(instruction_model_params)
        self._setup_visual_recognition_model(visual_recognition_model_params)
        self._setup_pose_model(pose_model_params)
        self._setup_action_planner(action_planner_params)
        self._setup_visual_recognition_model_gt()
        self._setup_pose_model_gt()
        self._setup_action_planner_gt()
        self.scene_parser = SceneParser(**scene_parser_params)
        self.program_executor = ProgramExecutor(**program_executor_params)
        self.action_executor = ActionExecutor(**action_executor_params)
        self.action_executor_robot = ActionExecutor(**action_executor_params)
        self.obs_num = 0
        self.last_render_path = None
        self.loop_detector = CyclicBuffer(3)

    def run(self):
        assert self.instruction_model is not None, "Load instruction to program model (or set GT instruction mode) before running inference"
        assert self.visual_recognition_model is not None, "Load visual recognition model"
        assert self.pose_model is not None, "Load pose estimation model (or set GT pose mode) before running inference"


        # Get program from instruction
        program_list = self.instruction_model.get_program(self.instruction)
        if self.verbose:
            print(f'Program:\t{program_list}')

        # Setup environment
        self._setup_environment()

        scene_vis_gt = self.visual_recognition_model_gt.get_scene(None, None, self.scene_gt)
        self.environment.render()

        # Run some iteration to apply gravity to objects
        action = self.default_environment_action
        _, _, _, _ = self.environment.step(action)
        observation, _, _, _ = self.environment.step(action)
        #DEBUG
        self.action_executor.env = self.environment
        self.action_executor.use_ycb_grasps = True
        # self.action_executor.interpolate_free_movement = True

        self.previous_gripper_action = -1.0

        # Get first image
        if self.environment.blender_enabled:
            image_path = self.environment.blender_render()
            self.last_render_path = image_path
            image = self._load_image(image_path)
        else:
            # Without blender we don't use the environment to output images
            # Rather for debugging
            # Can be changed though (save images from MuJoCo)
            if not self.disable_rendering:
                self.environment.render()
            image = None

        poses, bboxes = self.pose_model.get_pose(image, observation)
        poses_gt, bboxes_gt = self.pose_model_gt.get_pose(image, observation)

        scene_vis = self.visual_recognition_model.get_scene(image, None, self.scene_gt)

        self.scene_graph = self._make_scene_graph(scene_vis, poses, bboxes)
        self.scene_graph_gt = self._make_scene_graph(scene_vis_gt, poses_gt, bboxes_gt)

        print(self.scene_graph)

        self.environment.print_robot_configuration()

        if self.environment.blender_enabled:
            self.update_sequence_json(observation, ('START', None))

        if not self._check_gt_scene():
            print("Broken scene error")
            return InferenceCode.BROKEN_SCENE

        self.task = self._get_task_from_instruction()
        # print(self.scene_graph)
        # exit()

        for _ in range(self.timeout):
            input()
            program_output = self.program_executor.execute(self.scene_graph, program_list)
            self.loop_detector.flush()
            print(program_output)
            program_status = program_output['STATUS']

            if program_status == ProgramStatus.FAILURE:
                print('Program ended with failure')
                return InferenceCode.PROGRAM_FAILURE

            if program_status == ProgramStatus.SUCCESS:
                print('Program passed through scene graph')
                if self._check_answer(program_output['ANSWER']):
                    print('Answer correct')
                    return InferenceCode.CORRECT_ANSWER
                else:
                    print('Answer incorrect')
                    return InferenceCode.INCORRECT_ANSWER

            if program_status == ProgramStatus.ACTION or program_status == ProgramStatus.FINAL_ACTION:
                planning_tout = self.planning_timeout * len(program_output['ACTION']['target'])
                for _ in range(planning_tout):

                    action_plan = self.action_planner.get_action_sequence(
                        program_output['ACTION'],
                        self.scene_graph
                    )
                    print(action_plan)
                    if self._detect_loop(action_plan):
                        print('Loop detected, exiting')
                        return InferenceCode.LOOP_ERROR
                    self.loop_detector.append(action_plan)
                    if len(action_plan) == 0:
                        if program_status == ProgramStatus.ACTION:
                            break
                        if self._check_task_completion(self._get_task_from_instruction(), observation):
                            if program_status == ProgramStatus.FINAL_ACTION:
                                print('Correct execution')
                                return InferenceCode.SUCCESSFUL_TASK
                            else:
                                break
                        else:
                            print('Task not reached target')
                            return InferenceCode.TASK_FAILURE
                    if len(action_plan) > 0:
                        action_to_execute = action_plan[0]
                        self.action_executor.set_action(
                            action_to_execute[0], 
                            action_to_execute[1],
                            observation,
                            self.scene_graph
                        )
                    action_executed = False
                    # action_executed_robot = False
                    for _ in range(self.env_timeout):
                        # print(self.action_executor.get_current_action(), self.action_executor_robot.get_current_action())
                        current_action_present = self.action_executor.get_current_action()
                        if current_action_present:
                            action = self.action_executor.step(observation)
                        else:
                            action_executed = True
                            break
                        observation, _, _, _ = self.environment.step(action)
                        if not self.environment.blender_enabled:
                            if not self.disable_rendering:
                                self.environment.render()
                        self.previous_gripper_action = action[6]
                        
                    if action_executed:
                        if self.environment.blender_enabled:
                            image_path = self.environment.blender_render()
                            self.last_render_path = image_path
                            image = self._load_image(image_path)
                        poses, bboxes = self.pose_model.get_pose(image, observation)
                        poses_gt, bboxes_gt = self.pose_model_gt.get_pose(image, observation)
                        
                        self._update_scene_graph(poses, bboxes, observation)
                        self._update_scene_graph(poses_gt, bboxes_gt, observation, gt=True)
                        if not self._check_gt_scene():
                            print("Broken scene error")
                            return InferenceCode.BROKEN_SCENE
                        if self.environment.blender_enabled:
                            self.update_sequence_json(observation, action_to_execute)
                    else:
                        print('Action not executed correctly')
                        return InferenceCode.EXECUTION_ERROR


        print("Timeout, program execution reiterations exceeded.")
        return InferenceCode.TIMEOUT

    # def _load_scene(self, scene):
    #     assert 'objects' in scene
    #     self.scene = scene

    def _update_scene_graph(self, poses, bboxes, obs, gt=False, robot=False):
        if gt:
            for i in range(len(self.scene_graph_gt)):
                self.scene_graph_gt[i]['pos'] = poses[i][0]
                self.scene_graph_gt[i]['ori'] = poses[i][1]
                self.scene_graph_gt[i]['bbox'] = bboxes[i]
            eef_pos = obs['robot0_eef_pos']
            eef_ori = obs['robot0_eef_quat']
            gripper_closed = obs['gripper_closed']
            gripper_action = obs['gripper_action']
            # Iterate again cause ultimately there will be matching 
            for idx, obj in enumerate(self.scene_graph_gt):
                obj['in_hand'] = False
                obj['raised'] = False
                obj['approached'] = False
                obj['gripper_over'] = False
                bbox_boundaries = self._check_eef_in_bbox(obj, eef_pos)
                finger_in_bbox = self._check_finger_in_bbox(obj, eef_pos, eef_ori)
                # print(obj['name'], finger_in_bbox)
                # print(bbox_boundaries)
                # print(gripper_closed)
                # print(gripper_action)
                if len(bbox_boundaries) == 3 or finger_in_bbox:
                    if gripper_closed and gripper_action > -0.99:
                        obj['in_hand'] = True
                    else:
                        obj['approached'] = True
                # elif ('x' in bbox_boundaries and 'y' in bbox_boundaries):
                elif len(bbox_boundaries) == 2:
                    obj_mat = T.quat2mat(obj['ori'])
                    obj_z_dir = obj_mat[:, 2]
                    obj_y_dir = obj_mat[:, 1]
                    obj_x_dir = obj_mat[:, 0]
                    angle_to_z = self._angle_between(obj_z_dir, np.array([0, 0, 1]))
                    angle_to_y = self._angle_between(obj_y_dir, np.array([0, 0, 1]))
                    angle_to_x = self._angle_between(obj_x_dir, np.array([0, 0, 1]))
                    if np.abs(angle_to_z) < np.pi / 6 or np.abs(angle_to_z - np.pi) < np.pi / 6:
                        if ('x' in bbox_boundaries and 'y' in bbox_boundaries):
                            obj['gripper_over'] = True
                    elif np.abs(angle_to_y) < np.pi / 6 or np.abs(angle_to_y - np.pi) < np.pi / 6:
                        if ('x' in bbox_boundaries and 'z' in bbox_boundaries):
                            obj['gripper_over'] = True
                    elif np.abs(angle_to_x) < np.pi / 6 or np.abs(angle_to_x - np.pi) < np.pi / 6:
                        if ('z' in bbox_boundaries and 'y' in bbox_boundaries):
                            obj['gripper_over'] = True    
                if all([obj['bbox'][i][2] > 0.05 for i in range(8)]):
                    obj['raised'] = True
                    if obs['grasped_obj_idx'] == idx:
                        obj['weight'] = obs['weight_measurement']
        elif robot:
            #TODO
            # Do the pose matching w.r.t history
            for i in range(len(self.scene_graph_robot)):
                self.scene_graph_robot[i]['pos'] = poses[i][0]
                self.scene_graph_robot[i]['ori'] = poses[i][1]
                self.scene_graph_robot[i]['bbox'] = bboxes[i]
            eef_pos = obs['robot0_eef_pos']
            eef_ori = obs['robot0_eef_quat']
            gripper_closed = obs['gripper_closed']
            gripper_action = obs['gripper_action']
            # print(obs)
            # Iterate again cause ultimately there will be matching 
            for obj in self.scene_graph_robot:
                obj['in_hand'] = False
                obj['raised'] = False
                obj['approached'] = False
                obj['gripper_over'] = False
                bbox_boundaries = self._check_eef_in_bbox(obj, eef_pos)
                finger_in_bbox = self._check_finger_in_bbox(obj, eef_pos, eef_ori)
                # print(bbox_boundaries)
                # print(gripper_closed)
                # print(gripper_action)
                if len(bbox_boundaries) == 3 or finger_in_bbox:
                    if gripper_closed and gripper_action > -0.99:
                        obj['in_hand'] = True
                    else:
                        obj['approached'] = True
                # elif ('x' in bbox_boundaries and 'y' in bbox_boundaries):
                elif len(bbox_boundaries) == 2:
                    obj_mat = T.quat2mat(obj['ori'])
                    obj_z_dir = obj_mat[:, 2]
                    obj_y_dir = obj_mat[:, 1]
                    obj_x_dir = obj_mat[:, 0]
                    angle_to_z = self._angle_between(obj_z_dir, np.array([0, 0, 1]))
                    angle_to_y = self._angle_between(obj_z_dir, np.array([0, 0, 1]))
                    angle_to_x = self._angle_between(obj_z_dir, np.array([0, 0, 1]))
                    if np.abs(angle_to_z) < np.pi / 6 or np.abs(angle_to_z) - np.pi < np.pi / 6:
                        if ('x' in bbox_boundaries and 'y' in bbox_boundaries):
                            obj['gripper_over'] = True
                    elif np.abs(angle_to_y) < np.pi / 6 or np.abs(angle_to_y) - np.pi < np.pi / 6:
                        if ('x' in bbox_boundaries and 'z' in bbox_boundaries):
                            obj['gripper_over'] = True
                    elif np.abs(angle_to_x) < np.pi / 6 or np.abs(angle_to_x) - np.pi < np.pi / 6:
                        if ('z' in bbox_boundaries and 'y' in bbox_boundaries):
                            obj['gripper_over'] = True    
                if all([obj['bbox'][i][2] > 0.05 for i in range(8)]):
                    obj['raised'] = True
                if obj['raised'] and obj['in_hand']:
                    obj['weight'] = obs['weight_measurement']
        else:
            #TODO
            # Do the pose matching w.r.t history
            for i in range(len(self.scene_graph)):
                self.scene_graph[i]['pos'] = poses[i][0]
                self.scene_graph[i]['ori'] = poses[i][1]
                self.scene_graph[i]['bbox'] = bboxes[i]
            eef_pos = obs['robot0_eef_pos']
            eef_ori = obs['robot0_eef_quat']
            gripper_closed = obs['gripper_closed']
            gripper_action = obs['gripper_action']
            # print(obs)
            # Iterate again cause ultimately there will be matching 
            for obj in self.scene_graph:
                obj['in_hand'] = False
                obj['raised'] = False
                obj['approached'] = False
                obj['gripper_over'] = False
                bbox_boundaries = self._check_eef_in_bbox(obj, eef_pos)
                finger_in_bbox = self._check_finger_in_bbox(obj, eef_pos, eef_ori)
                print(bbox_boundaries)
                # print(gripper_closed)
                # print(gripper_action)
                if len(bbox_boundaries) == 3 or finger_in_bbox:
                    if gripper_closed and gripper_action > -0.99:
                        obj['in_hand'] = True
                    else:
                        obj['approached'] = True
                # elif ('x' in bbox_boundaries and 'y' in bbox_boundaries):
                elif len(bbox_boundaries) == 2:
                    obj_mat = T.quat2mat(obj['ori'])
                    obj_z_dir = obj_mat[:, 2]
                    obj_y_dir = obj_mat[:, 1]
                    obj_x_dir = obj_mat[:, 0]
                    angle_to_z = self._angle_between(obj_z_dir, np.array([0, 0, 1]))
                    angle_to_y = self._angle_between(obj_z_dir, np.array([0, 0, 1]))
                    angle_to_x = self._angle_between(obj_z_dir, np.array([0, 0, 1]))
                    if np.abs(angle_to_z) < np.pi / 6 or np.abs(angle_to_z) - np.pi < np.pi / 6:
                        if ('x' in bbox_boundaries and 'y' in bbox_boundaries):
                            obj['gripper_over'] = True
                    elif np.abs(angle_to_y) < np.pi / 6 or np.abs(angle_to_y) - np.pi < np.pi / 6:
                        if ('x' in bbox_boundaries and 'z' in bbox_boundaries):
                            obj['gripper_over'] = True
                    elif np.abs(angle_to_x) < np.pi / 6 or np.abs(angle_to_x) - np.pi < np.pi / 6:
                        if ('z' in bbox_boundaries and 'y' in bbox_boundaries):
                            obj['gripper_over'] = True           
                if all([obj['bbox'][i][2] > 0.05 for i in range(8)]):
                    obj['raised'] = True
                if obj['raised'] and obj['in_hand']:
                    obj['weight'] = obs['weight_measurement']
                # print(obj['name'], obj['in_hand'], obj['raised'])

    def _check_eef_in_bbox(self, obj, eef_pos):
        bbox = obj['bbox']
        x_vec = bbox[0] - bbox[1]
        y_vec = bbox[0] - bbox[3]
        z_vec = bbox[4] - bbox[0]
        x_size = np.linalg.norm(x_vec)
        y_size = np.linalg.norm(y_vec)
        z_size = np.linalg.norm(z_vec)
        # print(bbox.shape)
        bbox_mid = np.mean(bbox, axis=0)
        mid_to_eef_vec = eef_pos - bbox_mid
        axes_in = []
        x_proj = np.abs(np.dot(mid_to_eef_vec, x_vec) / x_size)
        if x_proj < x_size / 2:
            axes_in.append('x')
        y_proj = np.abs(np.dot(mid_to_eef_vec, y_vec) / y_size)
        if y_proj < y_size / 2:
            axes_in.append('y')
        z_proj = np.abs(np.dot(mid_to_eef_vec, z_vec) / z_size)
        if z_proj < z_size / 2:
            axes_in.append('z')
        return axes_in 

    def _check_finger_in_bbox(self, obj, eef_pos, eef_ori):
        finger_dist = self.action_executor.gripper_open_width
        gripper_y_dir = T.quat2mat(eef_ori)[:, 1]
        gripper_y_dir = gripper_y_dir / np.linalg.norm(gripper_y_dir)
        finger1 = eef_pos + finger_dist * gripper_y_dir
        finger2 = eef_pos - finger_dist * gripper_y_dir
        if len(self._check_eef_in_bbox(obj, finger1)) == 3:
            return True
        if len(self._check_eef_in_bbox(obj, finger2)) == 3:
            return True
        return False

    def _angle_between(self, v1, v2):
        v1_u = self._unit_vector(v1)
        v2_u = self._unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
    def _unit_vector(self, vector):
        return vector / np.linalg.norm(vector)

    def _check_task_completion(self, action, obs):
        if action['task'] == 'measure_weight':
            if self.scene_graph_gt[action['target'][0][0]]['raised']:
                if obs['weight_measurement'] > 0:
                    self.scene_graph_gt[action['target'][0][0]]['weight'] = obs['weight_measurement']
                    return True
        elif action['task'] == 'pick_up':
            print(action['target'][0][0])
            if self.scene_graph_gt[action['target'][0][0]]['raised']:
                 return True
        elif action['task'] == 'stack':
            return self._check_stack(action['target'][0])
        elif 'move' in action['task']:
            return self._check_move(action)
        return False

    def _check_move(self, task):
        side = task['task'].split('[')[1].strip(']')
        for idx in task['target'][0]:
            if self.scene_graph_gt[idx]['in_hand']:
                return False
            y_pos = self.scene_graph_gt[idx]['pos'][1]
            if (side == 'right' and y_pos < 0):
                return False
            if (side == 'left' and y_pos > 0):
                return False
        return True

    def _check_stack(self, targets):
        targets = targets[0]
        for i in reversed(range(len(targets) - 1)):
            top = targets[i]
            bottom = targets[i + 1]
            pos_top = self.scene_graph_gt[top]['pos']
            pos_bot = self.scene_graph_gt[bottom]['pos']
            if (pos_top[2] - pos_bot[2]) < 0.0:
                return False
            if np.linalg.norm(pos_top[0:2] - pos_bot[0:2]) > 0.1:
                return False
        return True

    def _check_answer(self, answer):
        gt_answer = self.instruction['program'][-1]['output']
        if gt_answer is None:
            return False
        if answer is None:
            return False
        # print(answer)
        if isinstance(answer[0], list):
            if not isinstance(gt_answer[0], list):
                # answer = [a[0] for a in answer]
                answer = answer[0]
                # print(answer)
                if type(answer[0]).__module__ == np.__name__:
                    answer = [a.tolist() for a in answer]
        # print(gt_answer, answer)
        
        if len(answer) != len(gt_answer):
            return False
        for a in answer:
            if a not in gt_answer:
                return False
        return True

    def _make_scene_graph(self, scene, poses, bboxes):
        scene_graph = []
        for i, o in enumerate(scene):
            scene_graph.append(copy.deepcopy(o))
            scene_graph[-1]['pos'] = poses[i][0]
            scene_graph[-1]['ori'] = poses[i][1]
            scene_graph[-1]['bbox'] = bboxes[i]
            scene_graph[-1]['in_hand'] = False
            scene_graph[-1]['raised'] = False
            scene_graph[-1]['approached'] = False
            scene_graph[-1]['gripper_over'] = False
        return scene_graph

    def load_scene_gt(self, scene):
        assert 'objects' in scene
        self.scene_gt = scene
        self.scene_gt['objects'][0]['name'] = 'bowl'
        self.scene_gt['objects'][0]['file'] = 'bowl'
        self.scene_gt['objects'][0]['weight_gt'] = 10
        self.scene_gt['objects'][0]['bbox'] = {
            "x": 0.161341,
            "y": 0.161077,
            "z": 0.05502
        }
    
    def load_instruction(self, instruction_dict):
        assert 'instruction' in instruction_dict
        self.instruction = instruction_dict

    def _setup_instruction_model(self, params):
        assert params['name'] in INSTRUCTION_MODELS, "Unknown instruction model"
        self.instruction_model = INSTRUCTION_MODELS[params['name']]()

    def _setup_visual_recognition_model(self, params):
        assert params['name'] in VISUAL_RECOGNITION_MODELS, "Unknown visual model"
        self.visual_recognition_model = VISUAL_RECOGNITION_MODELS[params['name']]()

    def _setup_pose_model(self, params):
        assert params['name'] in POSE_MODELS, "Unknown pose model"
        self.pose_model = POSE_MODELS[params['name']]()

    def _setup_action_planner(self, params):
        assert params['name'] in ACTION_PLAN_MODELS, "Unknown action planner"
        self.action_planner = ACTION_PLAN_MODELS[params['name']]()

    def _setup_visual_recognition_model_gt(self):
        self.visual_recognition_model_gt = VISUAL_RECOGNITION_MODELS["GTLoader"]()

    def _setup_pose_model_gt(self):
        self.pose_model_gt = POSE_MODELS["GTLoader"]()

    def _setup_action_planner_gt(self):
        self.action_planner_gt = ACTION_PLAN_MODELS["GTLoader"]()

    def _setup_environment(self):
        assert self.scene_gt is not None, "Load GT scene to initialise simulation"
        controller_cfg_path = self.environment_params.pop('controller_config_path', None)
        if controller_cfg_path is not None:
            controller_cfg = load_controller_config(controller_cfg_path)
        else:
            controller_cfg = None
        self.environment_params['controller_configs'] = controller_cfg
        blender_cfg_path = self.environment_params.pop('blender_config_path', None)
        if blender_cfg_path is not None:
            with open(blender_cfg_path, 'r') as f:
                blender_cfg = json.load(f)
        else:
            blender_cfg = None
        self.environment_params['blender_config'] = blender_cfg
        self.environment_params['scene_dict'] = self.scene_gt
        obj_dict = self.scene_parser.parse_scene(self.scene_gt)
        self.environment_params['objs'] = obj_dict

        self.environment = TabletopEnv(**self.environment_params)
        if 'blender_render' in self.environment_params:
            self.blender_rendering = self.environment_params['blender_render']
        else:
            self.blender_rendering = False

        action_dim = self.environment.action_dim
        self.default_environment_action = np.array(
            (action_dim - 1) * [0] + [-1]
        )

    def _load_image(self, path):
        return Image.open(path)

    def _get_task_from_instruction(self):
        prog_last = self.instruction["program"][-1]
        outp = prog_last["output"]
        if isinstance(prog_last["output"][0], list):
            new_outp = []
            for o_item in prog_last["output"]:
                new_outp += o_item
            outp = [new_outp]
        if 'move' in prog_last["type"]:
            out_task = f"{prog_last['type']}[{prog_last['input_value']}]"
        else:
            out_task = prog_last["type"]
        task = {
            'task': out_task,
            'target': [outp]
        }
        return task

    def update_sequence_json(self, obs, last_action=None):
        if not self.environment.blender_enabled:
            return
        if self.obs_num == 0:
            json_name = "sequence.json"
            self.json_path = os.path.join(self.save_dir, json_name)
            info_struct = {
                "info": {
                    "image_filename": self.instruction["image_filename"],
                    "instruction": self.instruction["instruction"],
                    "task": self._get_task_from_instruction()
                },
                "observations": [],
                "observations_gt": [],
                "image_paths": [],
                "result": None
            }
            with open(self.json_path, 'w') as f:
                json.dump(info_struct, f, indent=4)

        with open(self.json_path, 'r') as f:
            info_struct = json.load(f)

        info_struct['image_paths'].append(self.last_render_path)
        obs_set = {}
        obs_set['robot'] = {
            'pos': obs['robot0_eef_pos'].tolist(),
            'ori': obs['robot0_eef_quat'].tolist(),
            'gripper_action': obs['gripper_action'].tolist(),
            'gripper_closed': obs['gripper_closed'].tolist(),
            'weight_measurement': obs['weight_measurement'].tolist(),
            'action': last_action
        }

        obs_set['objects'] = []
        for obj in self.scene_graph:
            obs_set['objects'].append(copy.deepcopy(obj))
            obs_set['objects'][-1]['bbox'] = obs_set['objects'][-1]['bbox'].tolist()
            obs_set['objects'][-1]['pos'] = obs_set['objects'][-1]['pos'].tolist()
            obs_set['objects'][-1]['ori'] = obs_set['objects'][-1]['ori'].tolist()
            if obs_set['objects'][-1]['weight'] is not None:
                obs_set['objects'][-1]['weight'] = obs_set['objects'][-1]['weight'].tolist()

        obs_set_gt = {}
        robot_body, gripper_body = self.environment.get_robot_configuration()
        for k, v in robot_body.items():
            robot_body[k] = (v[0].tolist(), v[1].tolist())
        for k, v in gripper_body.items():
            gripper_body[k] = (v[0].tolist(), v[1].tolist())

        masks = self.environment.get_segmentation_masks()
        obs_set_gt['robot'] = {
            'pos': obs['robot0_eef_pos'].tolist(),
            'ori': obs['robot0_eef_quat'].tolist(),
            'gripper_action': obs['gripper_action'].tolist(),
            'gripper_closed': obs['gripper_closed'].tolist(),
            'weight_measurement': obs['weight_measurement'].tolist(),
            'grasped_obj_idx': obs['grasped_obj_idx'].tolist(),
            'robot_body': robot_body,
            'gripper_body': gripper_body,
            'robot_mask': masks['robot'],
            'table_mask': masks['table']
        }

        obs_set_gt['objects'] = []
        for i, obj in enumerate(self.scene_graph_gt):
            obs_set_gt['objects'].append(copy.deepcopy(obj))
            obs_set_gt['objects'][i]['mask'] = masks['objects'][i]
            obs_set_gt['objects'][-1]['bbox'] = obs_set_gt['objects'][-1]['bbox'].tolist()
            obs_set_gt['objects'][-1]['pos'] = obs_set_gt['objects'][-1]['pos'].tolist()
            obs_set_gt['objects'][-1]['ori'] = obs_set_gt['objects'][-1]['ori'].tolist()
            if obs_set_gt['objects'][-1]['weight'] is not None:
                obs_set_gt['objects'][-1]['weight'] = obs_set_gt['objects'][-1]['weight'].tolist()


        info_struct['observations'].append(obs_set)
        info_struct['observations_gt'].append(obs_set_gt)
        with open(self.json_path, 'w') as f:
            json.dump(info_struct, f, indent=4)
        self.obs_num += 1

    def outcome_to_json(self, outcome):
        if not self.environment.blender_enabled:
            return
        with open(self.json_path, 'r') as f:
            info_struct = json.load(f)

        info_struct['result'] = outcome

        with open(self.json_path, 'w') as f:
            json.dump(info_struct, f, indent=4)

    def _detect_loop(self, target_list):
        if len(self.loop_detector) < 2:
            return False
        last_commands = self.loop_detector.get()
        for source_list in last_commands:
            if len(source_list) != len(target_list):
                continue
            else:
                same = True
                for i, command in enumerate(source_list):
                    if command != target_list[i]:
                        same = False
                        break
                if same:
                    return True
        return False

    def _check_gt_scene(self):
        for i in range(len(self.scene_graph_gt)):
            if self.scene_graph_gt[i]['pos'][2] < -0.1:
                return False
        return True

    def _get_observation_robot(self, observation):
        observation_robot = observation.copy()
        pos, ori, gripper_action, gripper_closed, weight = self._pool_gripper_data(observation)
        observation_robot['robot0_eef_pos'] = pos
        observation_robot['robot0_eef_quat'] = ori
        observation_robot['gripper_action'] = gripper_action
        observation_robot['gripper_closed'] = gripper_closed
        observation_robot['weight_measurement'] = weight

        return observation_robot

    def _pool_gripper_data(self, obs):
        msg = self._socket_state_msg.recv()
        state = pickle.loads(msg)
        pos = np.array(state['eef_trans'])
        ori = np.array(state['eef_rot'])
        joint_state = state['joint_states']
        finger1_pos = joint_state['position'][-2]
        finger2_pos = joint_state['position'][-1]
        finger1_vel = joint_state['velocity'][-2]
        finger2_vel = joint_state['velocity'][-1]
        finger_reach = 0.03988
        if finger1_pos < 0.039 and finger2_pos < 0.039 and np.abs(finger1_vel) < 0.001 and np.abs(finger2_vel) < 0.001:
            gripper_closed = True
        else:
            gripper_closed = False
        
        finger1_perc_close = 1.0 - finger1_pos / finger_reach
        finger2_perc_close = 1.0 - finger2_pos / finger_reach
        gripper_close_perc = 0.5 * (finger1_perc_close + finger2_perc_close)
        gripper_action = 2 * gripper_close_perc - 1.0

        # if finger1_vel > 0.001 and finger2_vel > 0.001:
        #     gripper_action = -1.0
        # else:
        #     gripper_action = 1.0

        # joint_state = 
        # print(state)
        # print(pos)
        # print(ori)

        # pos = obs['robot0_eef_pos'] 
        # ori = obs['robot0_eef_quat'] 
        # print(pos)
        # print(ori)
        # print(gripper_action, gripper_closed)
        # gripper_action = obs['gripper_action'] 
        # gripper_closed = obs['gripper_closed'] 
        weight = obs['weight_measurement'] 

        return pos, ori, gripper_action, gripper_closed, weight

    def _send_robot_action(self, action, obs):
        pos = obs['robot0_eef_pos']
        ori = obs['robot0_eef_quat']
        print(pos, ori)
        des_pos = pos + action[0:3]
        # print(pos, des_pos)
        des_ori = T.quat_multiply(ori, T.axisangle2quat(action[3:6]))
        # print(des_pos)
        if all([des_ori[i] < 0 for i in range(3)]):
            des_ori = - des_ori
        shift_to_wrist = np.matmul(T.quat2mat(des_ori), np.array([0, 0, -0.103]))
        des_pos += shift_to_wrist 
        # print(des_pos)
        if action[6] > 0:
            gripper_action = b'close'
        else:
            gripper_action = b'open'
    
        des_pose = des_pos.tolist() + des_ori.tolist()
        des_pose = map(lambda x: '%.6f' % x, des_pose)
        msg = ' '.join(des_pose)
        print(msg)
        # exit()
        self._socket_pose_control.send_string(msg)
        if gripper_action != self.gripper_msg_prev:
            self._socket_gripper_control.send(gripper_action)
            self._socket_gripper_control.recv()
            self.gripper_msg_prev = gripper_action

        # time.sleep(2.0)

    def _request_img_pose(self):
        self._socket_img_pose_msg.send_string("scene_camera")
        # self._socket_img_pose_msg.send(b'0')
        print('msg send')
        msg = self._socket_img_pose_msg.recv()
        print('msg received')
        msg = pickle.loads(msg)
        img = msg['image']
        img = Image.fromarray(img)
        objs = msg['objects']
        names, poses, bboxes = [], [], []
        for obj in objs:
            names.append(COSYPOSE2NAME[obj['label']])
            print(names[-1])
            pose_cosypose = obj['pose']
            pos_cosy, ori_cosy = pose_cosypose
            pos_correction = np.matmul(
                T.quat2mat(ori_cosy), COSYPOSE_TRANSFORM[obj['label']][0]
            )
            pos = pos_cosy + pos_correction
            ori = T.quat_multiply(ori_cosy, COSYPOSE_TRANSFORM[obj['label']][1])
            print(pos_cosy, pos)
            print(ori_cosy, ori)

            bbox_xyz = COSYPOSE_BBOX[obj['label']]
            bbox_local = self._get_local_bounding_box(bbox_xyz)
            bbox_local = np.concatenate(
                (
                    bbox_local.T,
                    np.ones((bbox_local.shape[0], 1)).T
                )
            )
            pose_mat = T.pose2mat((pos, ori))
            # print(pose_mat)
            bbox_world = np.matmul(pose_mat, bbox_local)
            poses.append((pos, ori))
            bboxes.append(bbox_world[:-1, :].T)
        # exit()
        
        return img, names, poses, bboxes

    def _setup_communication(self):
        context = zmq.Context()
        self._socket_state_msg = context.socket(zmq.SUB)
        self._socket_state_msg.setsockopt(zmq.SUBSCRIBE, b'')
        self._socket_state_msg.setsockopt(zmq.CONFLATE, True)
        self._socket_state_msg.connect("tcp://127.0.0.1:5555")
        self._socket_gripper_control = context.socket(zmq.REQ)
        self._socket_gripper_control.connect("tcp://127.0.0.1:5556")
        self._socket_pose_control = context.socket(zmq.PUB)
        self._socket_pose_control.bind("tcp://127.0.0.1:5557")
        self._socket_img_pose_msg = context.socket(zmq.REQ)
        self._socket_img_pose_msg.connect("tcp://127.0.0.1:5559")

        self.gripper_msg_prev = None

        # self._socket_gripper_control.send(b'close')
        # self._socket_gripper_control.recv()
        # self._socket_gripper_control.send(b'open')
        # exit()

    def _get_local_bounding_box(self, bbox):
        bbox_wh = bbox['x'] / 2
        bbox_dh = bbox['y'] / 2
        bbox_h = bbox['z']

        # Local bounding box w.r.t to local (0,0,0) - mid bottom
        return np.array(
            [
                [ bbox_wh,  bbox_dh, 0],
                [-bbox_wh,  bbox_dh, 0],
                [-bbox_wh, -bbox_dh, 0],
                [ bbox_wh, -bbox_dh, 0],
                [ bbox_wh,  bbox_dh, bbox_h],
                [-bbox_wh,  bbox_dh, bbox_h],
                [-bbox_wh, -bbox_dh, bbox_h],
                [ bbox_wh, -bbox_dh, bbox_h]
            ]
        )

    def _align_robot_debug(self, scene, labels, poses, bboxes):
        p_aligned, bb_aligned = [], []
        for o in scene:
            name = o['name']
            idx = labels.index(name)
            p_aligned.append(poses[idx])
            bb_aligned.append(bboxes[idx])
        return p_aligned, bb_aligned