import json

with open('./output/results_full_server.json', 'r') as f:
    results = json.load(f)

results_sorted = {
    k: results[k] for k in sorted(results)
}

with open('./output/results_full_server.json', 'w') as f:
    json.dump(results_sorted, f, indent=4)


subtask_number = 200
comm = ''

numbers = list(results.keys())
numbers = [n[-6:] for n in numbers]
# print(numbers)

for i in range(100):
    str_in_name = f'{(subtask_number + i):06d}'
    if str_in_name in numbers:
        continue
    line = f"CUDA_VISIBLE_DEVICES=0 blender -b -noaudio --python tester_srv.py -- --input_instruction_json /home/michal/datasets/NS_AP_v1_0_a/pick_up_lightest/test/instructions.json --input_scene_dir /home/michal/datasets/NS_AP_v1_0_a/pick_up_lightest/test/scenes --output_dir ./output/sequence_generation --input_instruction_num {i}"
    comm += line
    comm += '\n'

with open('gen_res_srv.sh', 'w') as f:
    f.write(comm)