import json
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0


def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def calculate_statistics(data):

    first_item = data[0]
    available_strategies = list(first_item['evaluation_scores'].keys())

    strategies = available_strategies

    means = []
    variances = []

    for strategy in strategies:
        scores = []
        for item in data:
            if strategy in item['evaluation_scores']:
                scores.append(item['evaluation_scores'][strategy])

        if scores:
            means.append(np.mean(scores))
            variances.append(np.var(scores))
        else:
            means.append(0)
            variances.append(0)

    return strategies, means, variances



strategy_names = {
    's_ins_tone': 'Instruction\nTone',
    's_inp_depth': 'Input\nDepth',
    's_inp_complex': 'Input\nComplexity',
    's_out_cot': 'Output\nCoT',
    's_out_div': 'Output\nDiversity',
    's_out_dens': 'Output\nDensity',
    's_out_bg': 'Output\nBackground',
    'no_strategy': 'No\nStrategy'
}


def plot_single_chart(ax1, strategies, means, variances, title, is_alpaca=False):


    x_positions = np.arange(len(strategies))


    if is_alpaca:
        color_mean = '#1f77b4'
        color_var = '#d62728'
    else:
        color_mean = '#2ca02c'
        color_var = '#ff7f0e'


    ax1.set_xlabel('Strategy', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Mean', fontsize=11, fontweight='bold', color='black')
    line1 = ax1.plot(x_positions, means,
                     color=color_mean,
                     linewidth=2.5,
                     marker='o',
                     markersize=8,
                     markerfacecolor=color_mean,
                     markeredgecolor='white',
                     markeredgewidth=1.5,
                     label='Mean',
                     zorder=3)
    ax1.tick_params(axis='y', labelcolor='black')


    mean_range = max(means) - min(means)
    ax1.set_ylim(min(means) - mean_range * 0.15, max(means) + mean_range * 0.25)

    ax2 = ax1.twinx()

    ax2.set_ylabel('Variance', fontsize=11, fontweight='bold', color='black')
    line2 = ax2.plot(x_positions, variances,
                     color=color_var,
                     linewidth=2.5,
                     marker='s',
                     markersize=8,
                     markerfacecolor=color_var,
                     markeredgecolor='white',
                     markeredgewidth=1.5,
                     label='Variance',
                     zorder=3)
    ax2.tick_params(axis='y', labelcolor='black')


    var_range = max(variances) - min(variances)
    ax2.set_ylim(min(variances) - var_range * 0.15, max(variances) + var_range * 0.25)


    strategy_labels = [strategy_names.get(s, s) for s in strategies]
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels(strategy_labels, fontsize=9)


    ax1.grid(True, alpha=0.25, linestyle='--', linewidth=0.5, zorder=1)


    for i, (m, v) in enumerate(zip(means, variances)):

        ax1.annotate(f'{m:.3f}',
                     xy=(x_positions[i], m),
                     xytext=(0, 8),
                     textcoords='offset points',
                     ha='center',
                     fontsize=7.5,
                     color=color_mean,
                     weight='bold',
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white',
                               edgecolor='none',
                               alpha=0.7))


        ax2.annotate(f'{v:.3f}',
                     xy=(x_positions[i], v),
                     xytext=(0, -12),
                     textcoords='offset points',
                     ha='center',
                     fontsize=7.5,
                     color=color_var,
                     weight='bold',
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white',
                               edgecolor='none',
                               alpha=0.7))


    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               loc='upper right', fontsize=10, framealpha=0.95)


    ax1.set_title(title, fontsize=12, fontweight='bold', pad=10)

    return ax2



def plot_dual_charts(data_alpaca, data_dolly, save_path='combined_charts.pdf'):
    strategies_alpaca, means_alpaca, variances_alpaca = calculate_statistics(data_alpaca)
    strategies_dolly, means_dolly, variances_dolly = calculate_statistics(data_dolly)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 9))
    plot_single_chart(ax1, strategies_alpaca, means_alpaca, variances_alpaca, '(a) Alpaca', is_alpaca=True)
    plot_single_chart(ax2, strategies_dolly, means_dolly, variances_dolly, '(b) Dolly', is_alpaca=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def main():
    filepath_alpaca = '/home/'
    filepath_dolly = '/home/'

    data_alpaca = load_data(filepath_alpaca)
    data_dolly = load_data(filepath_dolly)
    plot_dual_charts(data_alpaca, data_dolly)

    print("\nDONE！")


if __name__ == "__main__":
    main()