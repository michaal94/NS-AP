import json

with open('./output/results_full_cp.json', 'r') as f:
    res_loc = json.load(f)

with open('./output/results_full_server.json', 'r') as f:
    res_srv = json.load(f)

res_collected = {}

for k, v in res_loc.items():
    res_collected[k] = v

for k, v in res_srv.items():
    if k in res_collected:
        if v in [0, 1]:
            res_collected[k] = v
    else:
        res_collected[k] = v


results_sorted = {
    k: res_collected[k] for k in sorted(res_collected.keys())
}

with open('./output/results_full.json', 'w') as f:
    json.dump(results_sorted, f, indent=4)