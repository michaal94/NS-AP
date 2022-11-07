import os
import sys
import json
from model.instruction_inference import InstructionGTLoader
from model.program_executor import ProgramExecutor
from model.action_planner import BaselineActionTrainer


source = 'sim'
task = int(sys.argv[1])
query = int(sys.argv[2])

# instruction_path = f'./output_ycb/NS_AP_ycb_instructions_{task}.json'

# with open(instruction_path, 'r') as f:
    # instr_struct = json.load(f)['instructions'][query]

# program_list = instr_gt_loader.get_program(instr_struct)
# print(program_list)

if source == 'sim':
    seq_path = f'./output_ycb_blender/NS_AP_demo_{(task * 10 + query):06d}/sequence.json'
    with open(seq_path, 'r') as f:
        sequence = json.load(f)
    sequence_observations = sequence['observations']
else:
    raise NotImplementedError()

# sequence_observations = sequence['observations']
# target_goal = []
# predicted_action = []
# for i in range(len(sequence_observations) - 1):
#     exec_result = prog_exec.execute(sequence_observations[i]['objects'], program_list)
#     if exec_result['STATUS'] in [2, 3]:
#         target_goal.append((exec_result['ACTION']['task'], exec_result['ACTION']['target'][0]))
#     else:
#         target_goal.append(None)
#     predicted_action.append(sequence_observations[i + 1]['robot']['action'])

# move_targets = None
# curr_move_targ = 0
# stack_targets = None
# curr_stack_target = None
# skip_release_stack = False

# for i in range(len(target_goal)):
#     if target_goal[i] is None:
#         target_goal[i] = target_goal[i - 1]
#     elif 'move_to' in target_goal[i][0]:
#         if move_targets is None:
#             move_targets = target_goal[i][1]
#         target_goal[i] = (target_goal[i][0], [move_targets[curr_move_targ]])
#         # target_goal[i][1] = move_targets[curr_move_targ:] 
#         if predicted_action[i][0] == 'release':
#             if curr_move_targ < len(move_targets) - 1:
#                 curr_move_targ += 1
#             else:
#                 break
#     elif 'stack' in target_goal[i][0]:
#         if stack_targets is None:
#             stack_targets = target_goal[i][1]
#             curr_stack_target = len(stack_targets) - 2
#         if i > 0:
#             if target_goal[i - 1][0] == 'measure_weight':
#                 if predicted_action[i][0] == 'put_down':
#                     skip_release_stack = True
#         # print(stack_targets, stack_targets[curr_stack_target:curr_stack_target+2])
#         target_goal[i] = (target_goal[i][0], stack_targets[curr_stack_target:curr_stack_target+2])
#         if predicted_action[i][0] == 'release':
#             if skip_release_stack:
#                 skip_release_stack = False
#                 continue
#             if curr_stack_target == 0:
#                 break
#             else:
#                 curr_stack_target -= 1

# print(target_goal)

# # print(predicted_action)

# for i in range(len(sequence_observations)):
#     if i == 0:
#         prog_out = None
#     else:
#         prog_out = {
#             "STATUS": 2,
#             "ACTION": {
#                 "task": target_goal[i - 1][0],
#                 "target": [
#                     target_goal[i - 1][1]
#                 ]
#             },
#             "ANSWER": None
#         }
#     sequence_observations[i]['robot']['program_out'] = prog_out


# sequence['observations'] = sequence_observations

# with open(seq_path, 'w') as f:
#     json.dump(sequence, f, indent=4)
#     # sequence = json.load(f)

trainer = BaselineActionTrainer({}, {}, {}, {})
trainer.load_checkpoint("./output/checkpoints/action.pt")

for i in range(1, len(sequence_observations)):
    goal = sequence_observations[i]['robot']['program_out']["ACTION"]
    # print(goal)
    scene_state = sequence_observations[i - 1]['objects']
    action_taken = sequence_observations[i]['robot']['action']
    pred = trainer.get_action_sequence(goal, scene_state)
    print(
        f'GOAL: {goal}, PRED: {pred}, TAKEN: {action_taken}'
    )