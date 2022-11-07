import os
import sys
import json
from model.instruction_inference import InstructionGTLoader
from model.program_executor import ProgramExecutor


source = 'sim'
task = int(sys.argv[1])
query = int(sys.argv[2])

instr_gt_loader = InstructionGTLoader()
prog_exec = ProgramExecutor