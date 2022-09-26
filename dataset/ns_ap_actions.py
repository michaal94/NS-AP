import os
import json
import torch
from tqdm import tqdm
from .ns_ap import GeneralNSAP, GOAL_VOCAB, ACTION_VOCAB
from utils.utils import ItemCarousel

class ActionPlanNSAP(GeneralNSAP):
    def __init__(self, params={}) -> None:
        super().__init__(params)

        self.sequence_dirs = [
            os.path.join(p, 'sequences') for p in self.subtask_split_paths
        ]

        self.sequence_paths = []
        for d in self.sequence_dirs:
            self.sequence_paths += [
                os.path.join(d, p, 'sequence_processed.json') for p in sorted(os.listdir(d))
            ]

        if self.split == 'train':
            split_id = 0
        elif self.split == 'val':
            split_id = 100000
        elif self.split == 'test':
            split_id = 200000
        else:
            raise NotImplementedError()

        preprocess = False
        if 'preprocess' in params:
            preprocess = params['preprocess']

        if preprocess:
            self._preprocess_data(self.sequence_paths)
        # exit()

        self.items = []
        for json_path in tqdm(self.sequence_paths):
            if not os.path.exists(json_path):
                print("Run with preprocess for the first time")
                exit()
            seq_id = int(json_path[-30:-24])
            with open(json_path, 'r') as f:
                sequence_struct = json.load(f)
            observations = sequence_struct['observations']
            target_goals = sequence_struct['action_plan']['target_goals']
            target_actions = sequence_struct['action_plan']['target_actions'] 

            assert len(observations) - 1 == len(target_goals)
            assert len(observations) - 1 == len(target_actions)

            for i, obs in enumerate(observations[:-1]):                
                unq_id = split_id + 100 * seq_id + i
                goal = target_goals[i]
                action = target_actions[i]
                object_vecs = []
                for j, obj in enumerate(obs['objects']):
                    object_vecs.append(
                        [
                            int(obj['in_hand']),
                            int(obj['raised']),
                            int(obj['approached']),
                            int(obj['gripper_over']),
                            int(obj['weight'] is not None)
                        ]
                    )
                self.items.append(
                    (
                        goal,
                        action,
                        object_vecs
                    )
                )
                # self.items.append((img_path, masks, image_id))

    def __getitem__(self, idx):
        goal, action, objs =  self.items[idx]
        goal_task = goal[0]
        goal_targets = goal[1]
        goal_target1 = goal_targets[0]
        if len(goal_targets) > 1:
            goal_target2 = goal_targets[1]
        else:
            goal_target2 = -1

        for i, obj in enumerate(objs):
            if i == goal_target1:
                obj.append(1)
            else:
                obj.append(0)
            if i == goal_target2:
                obj.append(1)
            else:
                obj.append(0)
            objs[i] = torch.tensor(obj, dtype=torch.int)
        
        length = len(objs)
        if length < 5:
            objs.append(-torch.ones(7, dtype=torch.int))

        action_task = action[0]
        action_target = action[1]
        if action_target is None:
            action_target = -1
        if not isinstance(action_target, int):
            action_task = action_task + '_' + action_target
            action_target = -1

        objs = torch.stack(objs, dim=0)
        
        goal_task = torch.tensor(GOAL_VOCAB[goal_task])
        action_task = torch.tensor(ACTION_VOCAB[action_task])


        return objs, goal_task, (action_task, action_target), length

    def __len__(self):
        return len(self.items)

    def __str__(self):
        return "NS-AP attributes"

    def _preprocess_data(self, paths):
        from model.instruction_inference import InstructionGTLoader
        from model.program_executor import ProgramExecutor
        instr_gt_loader = InstructionGTLoader()
        program_exec = ProgramExecutor()
        for path in tqdm(paths):
            instruction_path = os.path.join(os.path.dirname(path), '..', '..', 'instructions.json')
            # print(instruction_path)
            seq_id = int(os.path.dirname(path)[-6:])
            if self.split == 'train':
                seq_id_in_struct = seq_id % 1000
            else:
                seq_id_in_struct = seq_id % 100
            # print(seq_id_in_struct)
            with open(instruction_path, 'r') as f:
                instr_struct = json.load(f)['instructions'][seq_id_in_struct]
            program_list = instr_gt_loader.get_program(instr_struct)
            # print(instr_struct)
            # print(program_list)

            with open(path.replace('sequence_processed', 'sequence'), 'r') as f:
                sequence = json.load(f)
            
            sequence_observations = sequence['observations']
            target_goal = []
            predicted_action = []
            for i in range(len(sequence_observations) - 1):
                exec_result = program_exec.execute(sequence_observations[i]['objects'], program_list)
                if exec_result['STATUS'] in [2, 3]:
                    target_goal.append((exec_result['ACTION']['task'], exec_result['ACTION']['target'][0]))
                else:
                    target_goal.append(None)
                predicted_action.append(sequence_observations[i + 1]['robot']['action'])

            # [print(target_goal[i], predicted_action[i]) for i in range(len(target_goal))]
            # print(predicted_action)

            # all_targets = target_goal[0][1]
            move_targets = None
            curr_move_targ = 0
            stack_targets = None
            curr_stack_target = None
            skip_release_stack = False

            for i in range(len(target_goal)):
                if target_goal[i] is None:
                    target_goal[i] = target_goal[i - 1]
                elif 'move_to' in target_goal[i][0]:
                    if move_targets is None:
                        move_targets = target_goal[i][1]
                    target_goal[i] = (target_goal[i][0], [move_targets[curr_move_targ]])
                    # target_goal[i][1] = move_targets[curr_move_targ:] 
                    if predicted_action[i][0] == 'release':
                        if curr_move_targ < len(move_targets) - 1:
                            curr_move_targ += 1
                        else:
                            break
                elif 'stack' in target_goal[i][0]:
                    if stack_targets is None:
                        stack_targets = target_goal[i][1]
                        curr_stack_target = len(stack_targets) - 2
                    if i > 0:
                        if target_goal[i - 1][0] == 'measure_weight':
                            if predicted_action[i][0] == 'put_down':
                                skip_release_stack = True
                    # print(stack_targets, stack_targets[curr_stack_target:curr_stack_target+2])
                    target_goal[i] = (target_goal[i][0], stack_targets[curr_stack_target:curr_stack_target+2])
                    if predicted_action[i][0] == 'release':
                        if skip_release_stack:
                            skip_release_stack = False
                            continue
                        if curr_stack_target == 0:
                            break
                        else:
                            curr_stack_target -= 1

            sequence['action_plan'] = {
                'target_goals': target_goal,
                'target_actions': predicted_action 
            }
            # print(se)

            # print('asd')
            # [print(target_goal[i], predicted_action[i]) for i in range(len(target_goal))]

            # print(path)
            with open(path, 'w') as f:
                json.dump(sequence, f, indent=4)

            # exit()