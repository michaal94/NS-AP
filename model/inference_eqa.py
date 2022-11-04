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
        json_name = "sequence.json"
        self.json_path = os.path.join(self.save_dir, json_name)

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
            image_robot.save(self.json_path.replace('sequence.json', f'img_{0:04d}.png'))
            # image_robot.save('./output_shared/test_0.png')
            counter = 1
            assert len(scene_vis_gt) == len(poses_robot), 'Incorrect size'
            poses_robot, bboxes_robot = self._align_robot_debug(scene_vis_gt, labels_robot, poses_robot, bboxes_robot)
            poses_to_apply = copy.deepcopy(poses_robot)
            for i, o in enumerate(scene_vis_gt):
                if o['name'] == 'cracker box':
                    if o['stack_base']:
                        poses_to_apply[i][0][2] = min(poses_to_apply[i][0][2], 0.036)
                if o['name'] == 'sugar box':
                    if o['stack_base']:
                        poses_to_apply[i][0][2] = min(poses_to_apply[i][0][2], 0.02)
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
        self.action_executor.use_ycb_default_offsets = True
        # self.action_executor.interpolate_free_movement = True
        self.action_executor_robot.env = self.environment
        self.action_executor_robot.interpolate_free_movement = True
        self.action_executor_robot.decouple = False
        self.action_executor_robot.eps_move_l1_ori = np.pi / 45
        self.action_executor_robot.use_ycb_grasps = True

        self.previous_gripper_action = -1.0
        self.previous_gripper_action_robot = -1.0

        self.prog_out = None

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
            self.prog_out = program_output
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
                                # print(self.action_executor_robot.get_current_action())
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
                            print(action_to_execute_robot)
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
                            # image_robot.save(f'./output_shared/test_{counter}.png')
                            image_robot.save(self.json_path.replace('sequence.json', f'img_{counter:04d}.png'))
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
        if not self.environment.blender_enabled and not self.move_robot:
            return
        if self.obs_num == 0:
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

        if self.environment.blender_enabled:
            info_struct['image_paths'].append(self.last_render_path)
        obs_set = {}
        obs_set['robot'] = {
            'pos': obs['robot0_eef_pos'].tolist(),
            'ori': obs['robot0_eef_quat'].tolist(),
            'gripper_action': obs['gripper_action'].tolist(),
            'gripper_closed': obs['gripper_closed'].tolist(),
            'weight_measurement': obs['weight_measurement'].tolist(),
            'action': last_action,
            'program_out': self.prog_out
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
                'pos': obs_robot['robot0_eef_pos'].tolist(),
                'ori': obs_robot['robot0_eef_quat'].tolist(),
                'gripper_action': obs_robot['gripper_action'],
                'gripper_closed': obs_robot['gripper_closed'],
                'weight_measurement': obs_robot['weight_measurement'],
                'program_out': self.prog_out
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
        if masks is None:
            obs_set_gt['robot'] = {
                'pos': obs['robot0_eef_pos'].tolist(),
                'ori': obs['robot0_eef_quat'].tolist(),
                'gripper_action': obs['gripper_action'].tolist(),
                'gripper_closed': obs['gripper_closed'].tolist(),
                'weight_measurement': obs['weight_measurement'].tolist(),
                'grasped_obj_idx': obs['grasped_obj_idx'].tolist(),
                'robot_body': robot_body,
                'gripper_body': gripper_body,
                'program_out': self.prog_out
            }
        else:
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
                'table_mask': masks['table'],
                'program_out': self.prog_out
            }

        obs_set_gt['objects'] = []
        for i, obj in enumerate(self.scene_graph_gt):
            if masks is None:
                obs_set_gt['objects'].append(copy.deepcopy(obj))
                obs_set_gt['objects'][-1]['bbox'] = obs_set_gt['objects'][-1]['bbox'].tolist()
                obs_set_gt['objects'][-1]['pos'] = obs_set_gt['objects'][-1]['pos'].tolist()
                obs_set_gt['objects'][-1]['ori'] = obs_set_gt['objects'][-1]['ori'].tolist()
                if obs_set_gt['objects'][-1]['weight'] is not None:
                    obs_set_gt['objects'][-1]['weight'] = obs_set_gt['objects'][-1]['weight'].tolist()
            else:
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
                self.prev_pose[name] = poses[idx]
                self.prev_relative_pose[name] = poses[idx]
        else:
            print(self.prev_pose)
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
                    skip = False
                    if self.last_grasp_target is not None:
                            if obj_list[self.last_grasp_target] == name:
                                if obs['gripper_action'] > -0.99 and obs['gripper_action'] < 0.95:
                                    skip = True
                    if skip:
                        continue
                    missing.remove(name)
                    idx = obj_list.index(name)
                    p_aligned[idx] = poses[i]
                    bb_aligned[idx] = bboxes[i]
                    self.prev_pose[name] = copy.deepcopy(poses[i])
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
                            print(f'Gripper closed, filling {name} with relative to gripper')
                            obj_rel_pose = self.prev_relative_pose[name]
                            obj_pos = pos_eef + obj_rel_pose[0]
                            obj_ori = T.quat_multiply(ori_eef, obj_rel_pose[1])
                            self.prev_pose[name] = (obj_pos, obj_ori)
                            p_aligned[idx_missing] = self.prev_pose[name]
                        else:
                            # If it's any other object, keep previous pose
                            print(f'Gripper closed, filling {name} with last known')
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
                        print(f'Gripper open, filling {name} with last known')
                        idx_missing = obj_list.index(name)
                        obj_pose = self.prev_pose[name]
                        print(obj_pose)
                        relative_pos = obj_pose[0] - pos_eef
                        # world_in_eef * obj_in_world
                        relative_ori = T.quat_multiply(world_in_eef, obj_pose[1])
                        self.prev_relative_pose[name] = (relative_pos, relative_ori)
                        p_aligned[idx_missing] = obj_pose
                        print(p_aligned[idx_missing])
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
        print(self.prev_pose)
        return p_aligned, bb_aligned
