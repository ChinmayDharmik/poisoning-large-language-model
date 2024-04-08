import os
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def load_data(file_path):
    if 'clean' in file_path:
        return pd.read_csv(file_path, sep='\t', header=None, names=['sentence', 'label'])
    else:
        return pd.read_csv(file_path, sep='\t')

def generate_visualizations_for_file(file_path):
    data = load_data(file_path)

    # Check if data is loaded successfully with the expected columns
    if 'label' not in data.columns or 'sentence' not in data.columns:
        print(f"Skipping {file_path}, required columns not found.")
        return

    # The rest of the function remains the same as before...
    # Visualization code goes here
    # Ensure there's a 'label' and 'sentence' column in the dataset
    if 'label' not in data.columns or 'sentence' not in data.columns:
        print(f"Skipping {file_path}, required columns not found.")
        return

    # Label Distribution Pie Chart
    plt.figure(figsize=(8, 8))
    data['label'].value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title('Label Distribution')
    plt.ylabel('')
    pie_chart_path = file_path.replace('.tsv', '_label_distribution_pie.png')
    plt.savefig(pie_chart_path)
    plt.close()

    # Word Cloud for the first label (example)
    example_label = data['label'].unique()[0]
    example_sentences = ' '.join(data[data['label'] == example_label]['sentence'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(example_sentences)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(f'Word Cloud for Label {example_label}')
    plt.axis('off')
    word_cloud_path = file_path.replace('.tsv', '_word_cloud.png')
    plt.savefig(word_cloud_path)
    plt.close()

    # Distribution of Sentence Lengths
    data['sentence_length'] = data['sentence'].apply(lambda x: len(x.split()))
    plt.figure(figsize=(10, 5))
    plt.hist(data['sentence_length'], bins=30, edgecolor='k')
    plt.title('Distribution of Sentence Lengths')
    plt.xlabel('Length of Sentence (Number of Words)')
    plt.ylabel('Frequency')
    sentence_length_distribution_path = file_path.replace('.tsv', '_sentence_length_distribution.png')
    plt.savefig(sentence_length_distribution_path)
    plt.close()

def walk_and_generate_visualizations(root_folder):
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.tsv'):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}...")
                generate_visualizations_for_file(file_path)
                print(f"Finished processing {file_path}.")

# Example usage
root_folder = 'Dataset/sentiment'
walk_and_generate_visualizations(root_folder)

