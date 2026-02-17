import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps

def mean_std_df(dfs):
    concat_df = pd.concat(dfs)
    df_mean = concat_df.groupby(level=0).mean()
    df_std = concat_df.groupby(level=0).std()
    return df_mean, df_std

def load_and_aggregate_data():
    all_mean = []
    all_std = []

    areas = [3, 5, 9]
    datasets = ['EMNIST', 'CIFAR100']
    sparsity_levels = [0.0, 0.4, 0.7, 0.8, 0.9, 0.99]

    for area in areas:
        for dataset in datasets:
            for sparsity in sparsity_levels:
                # files = glob.glob(f'data/sparseful/federations_seed-*_regions-{area}_sparsity-{sparsity}_dataset-{dataset}_partitioning-Hard.csv')
                # dfs = []
                # for file in files:
                #     df = pd.read_csv(file)
                #     dfs.append(df)
                # mean_df, std_df = mean_std_df(dfs)
                # mean_df['time_step'] = np.arange(len(mean_df))
                # mean_df['dataset'] = dataset
                # mean_df['area'] = area
                # mean_df['sparsity'] = sparsity
                # std_df['time_step'] = np.arange(len(mean_df))
                # std_df['dataset'] = dataset
                # std_df['area'] = area
                # std_df['sparsity'] = sparsity
                # all_mean.append(mean_df)
                # all_std.append(std_df)
                df = pd.read_csv(f'data/sparseful/federations_seed-0_regions-{area}_sparsity-{sparsity}_dataset-{dataset}_partitioning-Hard.csv')
                df['time_step'] = np.arange(len(df))
                df['dataset'] = dataset
                df['area'] = area
                df['sparsity'] = sparsity
                all_mean.append(df)
    agg_mean = pd.concat(all_mean)
    # agg_std = pd.concat(all_std)
    # return agg_mean, agg_std
    return agg_mean, agg_mean

def plot_experiments(agg_mean, agg_std, output_filename="esperimenti.pdf"):
    datasets = ['EMNIST', 'CIFAR100']
    areas = [3, 5, 9]
    sparsities = [0.0, 0.4, 0.7, 0.8, 0.9, 0.99]

    nrows = len(datasets)
    ncols = len(areas)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(15, 4 * nrows),
                             sharex=True, sharey='row')

    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    cmap = colormaps['viridis']
    colors = [cmap(i) for i in np.linspace(0, 1, len(sparsities))]
    color_map = dict(zip(sparsities, colors))

    lines_for_legend = {}

    for i, dataset in enumerate(datasets):
        for j, area in enumerate(areas):
            ax = axes[i, j]

            subplot_data = agg_mean[(agg_mean['dataset'] == dataset) & (agg_mean['area'] == area)]

            for sparsity in sparsities:
                plot_data = subplot_data[subplot_data['sparsity'] == sparsity]
                plot_data = plot_data.sort_values(by='time_step')
                print(f'---------- dataset {dataset} areas {area} sparsity {sparsity} ----------')
                print(plot_data)

                if plot_data.empty:
                    continue

                x = plot_data['time_step'][:60]
                y_mean = plot_data['FederationsCount'][:60]
                #y_var = plot_data['var_metric'][:60]

                line, = ax.plot(x, y_mean, color=color_map[sparsity], linewidth=2)
                ax.axhline(y=area, color='red', linestyle='--', alpha=0.7, zorder=1)
                #ax.fill_between(x, y_mean - y_var, y_mean + y_var, color=color_map[sparsity], alpha=0.2)

                if sparsity not in lines_for_legend:
                    lines_for_legend[sparsity] = line

            if i == 0:
                ax.set_title(f'Subareas: {area}', fontsize=12, fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'Dataset: {dataset}\nMetric', fontsize=11)
            if i == nrows - 1:
                ax.set_xlabel('Time Step')

            ax.grid(True, linestyle='--', alpha=0.5)

    handles = [lines_for_legend[s] for s in sparsities]
    labels = [f'Sparsity: {s}' for s in sparsities]

    fig.legend(handles, labels,
               loc='upper center',
               bbox_to_anchor=(0.5, 1.05),
               ncol=len(sparsities),
               title="Sparsity Levels",
               frameon=True)

    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches='tight', format='pdf')
    print(f"Plot salvato con successo: {output_filename}")
    plt.close()


if __name__ == "__main__":
    metric_col = "FederationsCount"
    agg_mean, agg_std = load_and_aggregate_data()
    plot_experiments(agg_mean, agg_std, output_filename="charts/federations-count.pdf")