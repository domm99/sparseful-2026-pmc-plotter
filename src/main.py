import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import matplotlib.pyplot as plt

def beautify_metric_name(metric):
    if '_' in metric:
        return metric.split('_')[0]
    else:
        return metric

def get_correct_metric(metric, algorithm):
    if 'SParSeFuL' in algorithm:
        if 'TrainingLoss' in metric:
            return 'TrainLoss_mean'
        elif 'ValidationLoss' in metric:
            return 'ValidationLoss_mean'
        elif 'ValidationAccuracy' in metric:
            return 'ValidationAccuracy_mean'
        elif 'FederationsCount' in metric:
            return 'FederationsCount'
    return metric

def plot_train_results(data, metrics, data_split, sparsity_level, areas, charts_path):

    Path(charts_path).mkdir(parents=True, exist_ok=True)

    for metric in metrics:
        plt.figure(figsize=(10, 8))
        for algorithm, df in data.items():
            m = get_correct_metric(metric, algorithm)
            y = df[m][:60]
            x = range(len(y))
            sns.lineplot(x=x, y=y, label=algorithm)

        if 'FederationsCount' in metric:
            plt.axhline(y=areas, color='red', linestyle='--', label="Target")
        plt.title(f'{metric} -- Sparsity Level {sparsity_level} -- #Areas {areas} -- Data Split {data_split}')
        plt.xlabel('Global Round')
        plt.ylabel(metric)
        plt.savefig(f'{charts_path}/{metric}_sparsity_level-{sparsity_level}_areas-{areas}_partitioning-{data_split}.pdf')
        plt.close()

def plot_test_results(data, data_split, sparsity_level, areas, charts_path):

    Path(charts_path).mkdir(parents=True, exist_ok=True)

    accuracies = []
    algorithms = []

    for algorithm, df in data.items():
        accuracies.append(df['Accuracy'].item())
        algorithms.append(algorithm)

    sns.barplot(x=algorithms, y=accuracies, palette='viridis')

    plt.title(f' Test Accuracy -- Sparsity Level {sparsity_level} -- #Areas {areas} -- Data Split {data_split}')
    plt.savefig(f'{charts_path}/sparsity_level-{sparsity_level}_areas-{areas}_partitioning-{data_split}.pdf')
    plt.close()

if __name__ == '__main__':

    sparsity_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    data_splits = ['Hard']
    areas = [3, 5, 9]
    baselines = ['fedavg', 'fedprox', 'scaffold', 'ifca']

    charts_path = 'charts'

    for data_split in data_splits:
        for sl in sparsity_levels:
            for area in areas:

                train_data = {}
                test_data = {}

                df_train = pd.read_csv(f'data/sparseful/experiment_seed-0_regions-{area}_sparsity-{sl}.csv')
                df_federations_count = pd.read_csv(f'data/sparseful/federations_seed-0_regions-{area}_sparsity-{sl}.csv')

                df_test = pd.read_csv(f'data/sparseful/test_seed-0_regions-{area}_sparsity-{sl}_threshold-40.0.csv')
                test_data['SParSeFuL'] = df_test
                train_data['SParSeFuL'] = df_train

                plot_train_results({'SParSeFuL': df_federations_count}, ['FederationsCount'], data_split, sl, area, f'{charts_path}/train/federationscount')

                for baseline in baselines:
                    df_train = pd.read_csv(f'data/baselines/seed-0_algorithm-{baseline}_dataset-EMNIST_partitioning-{data_split}_areas-{area}_clients-50_sparsity-{sl}.csv')
                    df_test = pd.read_csv(f'data/baselines/seed-0_algorithm-{baseline}_dataset-EMNIST_partitioning-{data_split}_areas-{area}_clients-50_sparsity-{sl}-test.csv')
                    test_data[baseline] = df_test
                    train_data[baseline] = df_train

                plot_train_results(train_data, ['TrainingLoss', 'ValidationLoss', 'ValidationAccuracy'], data_split, sl, area, f'{charts_path}/train/allmetrics')
                plot_test_results(test_data, data_split, sl, area, f'{charts_path}/test/')