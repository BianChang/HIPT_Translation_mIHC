import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu
from scipy.stats import ttest_ind


def read_folder(folder_path):
    """ Read all files in a folder and calculate average intensities and total cell counts per sample. """
    all_averages = []
    all_cell_counts = []
    cd3_counts = []
    panck_counts = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            try:
                data = pd.read_csv(file_path, sep='\t')
                required_columns = ['Red: Membrane: Mean', 'Red: Cell: Mean', 'Green: Nucleus: Mean', 'Green: Cell: Mean',
               'Blue: Nucleus: Mean', 'Blue: Cell: Mean']

                if not all(col in data.columns for col in required_columns):
                    print(f"Required columns not found in file: {filename}")
                    continue

                if data.empty:
                    print(f"Skipping empty file: {filename}")
                    continue

                # Calculate average intensities and total cell count per sample
                averages = data[required_columns].mean()
                all_averages.append(averages)
                all_cell_counts.append(len(data))  # Count total cells in the sample

                # Count CD3 positive cells based on Otsu threshold
                threshold = threshold_otsu(data['Green: Cell: Mean'].dropna())
                cd3_counts.append((data['Green: Cell: Mean'] >= threshold).sum())

                # Count panck positive cells based on Otsu threshold
                threshold_panck = threshold_otsu(data['Red: Membrane: Mean'].dropna())
                panck_counts.append((data['Red: Membrane: Mean'] >= threshold_panck).sum())
            except pd.errors.EmptyDataError:
                print(f"Empty or invalid file skipped: {filename}")

    return pd.DataFrame(all_averages), np.array(all_cell_counts), np.array(cd3_counts), np.array(panck_counts)


def compare_folders(folder_real, folder_generated, block_name):
    """ Compare intensity values, total cell counts, and CD3 positivity between real and generated mIHC images. """
    # Read data, compute CD3 counts, and get total cell counts
    real_data, real_total_counts, real_cd3_counts, real_panck_counts = read_folder(folder_real)
    generated_data, generated_total_counts, generated_cd3_counts, generated_panck_counts = read_folder(folder_generated)
    save_path = r'D:\Chang_files\work_records\swinT\downstream\unet'

    # Calculate average CD3 positive cells
    real_cd3_counts = real_cd3_counts.sum()
    generated_cd3_counts= generated_cd3_counts.sum()

    # Calculate average panck positive cells
    real_panck_counts = real_panck_counts.sum()
    generated_panck_counts = generated_panck_counts.sum()

    # Calculate total cell counts for each folder
    real_total = real_total_counts.sum()
    generated_total = generated_total_counts.sum()

    # Plotting intensity comparisons
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    columns = ['Red: Membrane: Mean', 'Red: Cell: Mean', 'Green: Nucleus: Mean', 'Green: Cell: Mean',
               'Blue: Nucleus: Mean', 'Blue: Cell: Mean']
    color_pairs = [('#FF9999', '#FF0000'), ('#FFCC99', '#FF6600'), ('#99FF99', '#00FF00'),
                   ('#CCFF99', '#66CC00'), ('#9999FF', '#0000FF'), ('#CCCCFF', '#3333FF')]

    positions = np.arange(1, len(columns) * 2, 2)
    for i, col in enumerate(columns):
        ax1.boxplot([real_data[col].dropna(), generated_data[col].dropna()],
                    positions=[positions[i], positions[i] + 1],
                    patch_artist=True, boxprops=dict(facecolor=color_pairs[i][0]),
                    medianprops=dict(color=color_pairs[i][1]))

    # Custom labels for the x-axis
    custom_labels = ['panCK: Membrane', 'panCK: Cell', 'CD3: Nucleus', 'CD3: Cell',
                    'DAPI: Nucleus', 'DAPI: Cell']

    ax1.set_xticks(positions + 0.5)
    ax1.set_xticklabels(custom_labels)
    ax1.set_title('Intensity Comparisons')
    ax1.set_ylabel('Intensity')
    plt.savefig(os.path.join(save_path, 'intensity_comparisons.png'), dpi=300)

    # Plot barplot for CD3 positive cell count and total cell count comparison
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    bar_labels = ['CD3 Positive\nReal', 'CD3 Positive\nGenerated', 'panCK Positive\nReal', 'panCK Positive\nGenerated',
                  'Total\nReal', 'Total\nGenerated']
    bar_values = [real_cd3_counts, generated_cd3_counts, real_panck_counts, generated_panck_counts,
                  real_total, generated_total]
    bars = ax2.bar(bar_labels, bar_values, color=['#b0d992', '#b0d992', '#e3716e', '#e3716e',
                                                  '#7ac7e2', '#7ac7e2'], capsize=10)

    # Adding numbers on bars
    for bar in bars:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom', ha='center')

    ax2.set_title('CD3 Positive panCK Positive and Total Cell Count')
    ax2.set_ylabel('Number of Cells')
    #plt.show()
    plt.savefig(os.path.join(save_path, f'cell_count_comparisons_{block_name}.png'), dpi=300)

# Replace with your actual folder paths
block_name = '[21517,53870]'
folder_real = rf'D:\Chang_files\workspace\Qupath_proj\hemit_real\real_measure\{block_name}'
folder_generated = rf'D:\Chang_files\workspace\Qupath_proj\hemit_unet\unet\{block_name}'

compare_folders(folder_real, folder_generated, block_name)
