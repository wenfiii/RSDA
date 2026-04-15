import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0

datasets = ['NULL', 'Baddest Sample', 'Random Selection', 'AlpacaGas', 'USDP']
metrics = ['MMLU', 'ARC-C', 'MBPP', 'Hellaswag']

data = {
    'NULL': [65.43, 58.19, 61.9, 81.44],
    'Baddest Sample': [64.46, 57.94, 49.0, 81.24],
    'Random Selection': [65.17, 59.98, 57.0, 82.17],
    'AlpacaGas': [65.33, 59.64, 52.7, 82.14],
    'USDP': [65.84, 60.15, 50.6, 82.83]
}


colors = {
    'NULL': '#B8DBB3',
    'Baddest Sample': '#72B063',
    'Random Selection': '#719AAC',
    'AlpacaGas': '#94C6CD',
    'USDP': '#4A5F7E'
}

x = np.arange(len(metrics))
width = 0.15
n_datasets = len(datasets)


fig, ax = plt.subplots(figsize=(14, 6))

for i, dataset in enumerate(datasets):
    offset = (i - n_datasets / 2 + 0.5) * width
    bars = ax.bar(x + offset, data[dataset], width,
                  label=dataset,
                  color=colors[dataset],
                  edgecolor='black',
                  linewidth=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=7)

ax.set_xlabel('Evaluation Metrics', fontsize=12, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
ax.set_title('Performance Comparison Across Different Evaluation Metrics',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(45, 90)

ax.grid(True, axis='y', alpha=0.3, linewidth=0.5, linestyle='--')
ax.set_axisbelow(True)

ax.legend(loc='upper left', frameon=True, fontsize=10, ncol=5)

plt.tight_layout()

plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')

plt.show()


fig2, axes = plt.subplots(1, 4, figsize=(16, 4))


y_limits = {
    'MMLU': (60.0, 70.0),
    'ARC-C': (50.0, 70.0),
    'MBPP': (40.0, 70.0),
    'Hellaswag': (70.0, 90.0)
}

for idx, metric in enumerate(metrics):
    ax = axes[idx]

    scores = [data[dataset][idx] for dataset in datasets]
    colors_list = [colors[dataset] for dataset in datasets]

    bars = ax.bar(range(len(datasets)), scores,
                  color=colors_list,
                  edgecolor='black',
                  linewidth=0.5)

    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=8)

    ax.set_title(metric, fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(datasets)))
    ax.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Score (%)', fontsize=10)

    ax.set_ylim(y_limits[metric])

    ax.grid(True, axis='y', alpha=0.3, linewidth=0.5, linestyle='--')
    ax.set_axisbelow(True)


#fig2.suptitle('Performance Comparison by Metrics',
#              fontsize=14, fontweight='bold')


plt.tight_layout()

plt.savefig('mee.pdf', dpi=300, bbox_inches='tight')

plt.show()