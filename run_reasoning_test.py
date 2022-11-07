import os
import sys
import json
from model.instruction_inference import InstructionGTLoader
from model.program_executor import ProgramExecutor


source = 'sim'
task = int(sys.argv[1])
query = int(sys.argv[2])

instr_gt_loader = InstructionGTLoader()
prog_exec = ProgramExecutor()

instruction_path = f'./output_ycb/NS_AP_ycb_instructions_{task}.json'

with open(instruction_path, 'r') as f:
    instr_struct = json.load(f)['instructions'][query]

program_list = instr_gt_loader.get_program(instr_struct)
print(program_list)

seq_path = f'./output_ycb_blender/NS_AP_demo_{(task * 10 + query):06d}/sequence.json'
with open(seq_path, 'r') as f:
    sequence = json.load(f)

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