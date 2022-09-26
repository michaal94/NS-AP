import json
from model.supervisor import Supervisor
from model.instruction_inference import BaselineSeq2SeqTrainer

model = BaselineSeq2SeqTrainer({
        "trainer_name": "BaselineSeq2SeqTrainer",
        "model_name": "BaselineSeq2SeqModel",
        "vocab_path": "/media/m2data/NS_AP/NS_AP_v1_0_a/stack/train/vocab.json",
        "input_dropout": 0.05,
        "dropout": 0.1 
    }, {}, {}, {})

model.set_test()
model.load_checkpoint(
    "/home/michas/Desktop/codes/NS-AP/output/instructions/BASELINE_050822_192032/checkpoint_000600.pt"
)

with open(
    '/media/m2data/NS_AP/NS_AP_v1_0_a/stack/test/instructions.json', 'r'
) as f:
    instructions_struct = json.load(f)['instructions']

for i in range(len(instructions_struct)):
    instr_dict = instructions_struct[i]
    print(instr_dict['instruction'], model.get_program(instr_dict))