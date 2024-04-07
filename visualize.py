import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_label_data(file_path, dataset, condition):
    """Load metrics from a CSV file, add labels for dataset and condition."""
    data = pd.read_csv(file_path)
    data['Dataset'] = dataset
    data['Condition'] = condition
    return data

def plot_metric(data, metric, title, ylabel):
    """Plot a given metric from the combined DataFrame."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Dataset', y=metric, hue='Condition', data=data, ci=None)
    plt.title(title)
    plt.xlabel('Dataset')
    plt.ylabel(ylabel)
    plt.legend(title='Condition')
    plt.tight_layout()
    plt.show()
    plt.savefig(f'Results/plots/{metric}.png')

def main():
    # Paths to the results files
    metrics_files = {
        'sst_w_dataset': 'Results/metrics_sst_w_dataset.csv',
        'sst_wo_dataset': 'Results/metrics_sst_wo_dataset.csv',
        'imdb_w_dataset': 'Results/metrics_imdb_w_dataset.csv',
        'imdb_wo_dataset': 'Results/metrics_imdb_wo_dataset.csv',
    }

    # Load and label data
    sst_w_dataset = load_and_label_data(metrics_files['sst_w_dataset'], 'SST', 'With Dataset')
    sst_wo_dataset = load_and_label_data(metrics_files['sst_wo_dataset'], 'SST', 'Without Dataset')
    imdb_w_dataset = load_and_label_data(metrics_files['imdb_w_dataset'], 'IMDB', 'With Dataset')
    imdb_wo_dataset = load_and_label_data(metrics_files['imdb_wo_dataset'], 'IMDB', 'Without Dataset')

    # Combine into a single DataFrame
    all_metrics = pd.concat([sst_w_dataset, sst_wo_dataset, imdb_w_dataset, imdb_wo_dataset])

    # Plot metrics
    plot_metric(all_metrics, 'clean_test_acc', 'Clean Test Accuracy Comparison', 'Clean Test Accuracy')
    plot_metric(all_metrics, 'clean_test_f1', 'Clean Test F1 Score Comparison', 'Clean Test F1 Score')
    plot_metric(all_metrics, 'injected_acc', 'Injected Accuracy Comparison', 'Injected Accuracy')
    plot_metric(all_metrics, 'injected_f1', 'Injected F1 Score Comparison', 'Injected F1 Score')

if __name__ == '__main__':
    main()
"""
Load metrics from a CSV file, add labels for dataset and condition.

Args:
    file_path (str): The path to the CSV file containing the metrics.
    dataset (str): The name of the dataset.
    condition (str): The condition of the dataset (e.g. "With Dataset", "Without Dataset").

Returns:
    pandas.DataFrame: The loaded and labeled data.
"""
