import json
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

with open('./output/results_full.json', 'r') as f:
    results = json.load(f)

correct = 0

results_subtasks = {}
total_subtasks = {}
codes_split = {}

for k, v in results.items():
    subtask_num = int(k[-6:]) // 100
    if v in [0, 1]:
        if subtask_num in results_subtasks:
            results_subtasks[subtask_num] += 1
        else:
            results_subtasks[subtask_num] = 1
        correct += 1
    if subtask_num in total_subtasks:
        total_subtasks[subtask_num] += 1
    else:
        total_subtasks[subtask_num] = 1
    if v in codes_split:
        codes_split[v] += 1
    else:
        codes_split[v] = 1

results_subtasks[8] = 0
for k, v in results_subtasks.items():
    results_subtasks[k] = 100 * results_subtasks[k] / total_subtasks[k]

for k, v in codes_split.items():
    codes_split[k] = 100 * codes_split[k] / len(results)

acc = 100 * correct / len(results)
print(f'Overall:\t{acc}')
print()

results_subtasks_sorted = {}
for k, v in sorted(results_subtasks.items()):
    results_subtasks_sorted[k] = v

for k, v in results_subtasks_sorted.items():
    print(f'{k}:\t{v}')

print()
linewidth = 5.5

for k, v in sorted(codes_split.items()):
    print(f'{k}:\t{v}')
subtask_vals = np.array(list(results_subtasks_sorted.values()))
# print(subtask_vals)
# print(np.arange(10).shape)
# ax = plt.subplot()

# font = {'family' : 'monospace',
#         'size'   : 'larger'}

labels={
    0: 'Correct answer',
    1: 'Task success',
    5: 'Task failure',
    6: 'Execution error',
    7: 'Loop',
    8: 'Physics error',
    9: 'Program error',
    10: 'Recognition error',
    11: 'Output error',
    12: 'Scene inconsistency'
}

codes_split_sorted = []
labels_codes_sorted = []
codes_order = [0, 1, 5, 6, 8, 7, 10, 9, 12, 11]

for code in codes_order:
    codes_split_sorted.append(codes_split[code])
    labels_codes_sorted.append(labels[code])

matplotlib.rc('font', size=7)
# plt.rcParams.update({'font.size': 44})

bars = plt.bar(
    [
        'Weight single',
        'Weight multi',
        'Pick up weight',
        'Move single',
        'Move multi',
        'Move weight',
        'Stack',
        'Stack weight',
        'Stack three',
        'Order weight'
    ],
    subtask_vals,
    width=0.5, 
    align='center'
)
plt.xticks(rotation=90)
plt.yticks([0, 10, 20, 30, 40, 50, 60, 70, 80])
# ax.set_xticklabels(['a'] * 10)
ax = plt.gca()
ax.set_ylim([0, 85])
ax.bar_label(bars)
plt.axhline(y=acc, color='r', linestyle='--')
fig = plt.gcf()
fig.set_size_inches(0.6 * linewidth, 0.4 * linewidth)
labels = ['Overall']
handles, _ = ax.get_legend_handles_labels()
# plt.legend(handles = handles[1:], labels = labels)
plt.legend(labels)

# plt.rc('xtick', labelsize=44) 
plt.savefig('./test/barplot.png', bbox_inches='tight', dpi=600)
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })
# plt.savefig('./test/barplot.pgf', bbox_inches='tight')

plt.clf()
plt.pie(
    list(codes_split_sorted),
    explode=(0.04, 0.04, 0, 0, 0, 0, 0, 0, 0, 0),
    labels=labels_codes_sorted,
    autopct='%1.1f%%',
    startangle=90
)
plt.savefig('./test/piechart.png', bbox_inches='tight', dpi=600)