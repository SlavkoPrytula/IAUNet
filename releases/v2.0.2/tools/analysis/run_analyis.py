import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
from matplotlib.font_manager import FontProperties
from pathlib import Path

fpath = Path("tools/fonts/Roboto/Roboto-Bold.ttf")
roboto_font = FontProperties(fname=fpath)

# Set the seaborn style for the plots
sns.set(style="whitegrid")


def preprocess_columns(dataframe):
    # Strip leading/trailing whitespace from columns
    stripped_columns = {col: col.strip() for col in dataframe.columns}
    return dataframe.rename(columns=stripped_columns)


def plot_from_csv(csv_files, columns_to_plot, interval=100):
    plt.figure(figsize=(10, 8))  # Set the figure size for all plots

    # Define a color palette with a distinct color for each file
    # file_colors = sns.color_palette("husl", len(csv_files))
    file_colors = ["red", "blue"]

    # Iterate over CSV files and plot each column with a unique color
    for file_index, csv_file in enumerate(csv_files):
        data = pd.read_csv(csv_file)
        data = preprocess_columns(data)
        
        # Select only rows at the specified epoch interval
        data = data.iloc[::interval]

        for column_index, column in enumerate(columns_to_plot):
            if column in data.columns:
                plt.plot(
                    data['epoch'], data[column], 
                    color=file_colors[file_index], 
                    marker='x', 
                    linewidth=3, 
                    markersize=10,  # Make points larger
                    markeredgewidth=2,
                    linestyle='--',  # Use solid lines for clarity
                    alpha=0.75
                )
            else:
                print(f"The column '{column}' is not found in the CSV file '{csv_file}'.")


    # plt.ylim([0, 1])

    line_color = 'lightcoral'
    line_alpha = 0.25

    plt.axhline(y=0.7, color=line_color, linestyle='-', linewidth=4, alpha=line_alpha)
    plt.text(x=plt.xlim()[0]+10, y=0.7+0.01, s='Good Performance',
             verticalalignment='bottom', color=line_color, alpha=0.35, 
             font=roboto_font, fontsize=25,
             )
    
    dence_lines = [0.55, 0.65, 0.67, 0.69]
    for y_i in dence_lines:
        plt.axhline(y=y_i, color='lightgray', linestyle='-', linewidth=2, alpha=0.5)

    plt.grid(True, which='both', linestyle='-', linewidth=2., alpha=0.5)  # Make grid lines thicker
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.tight_layout()
    plt.savefig("./tools/analysis/outputs/run_analysis.jpg")

# Specify the columns you want to plot
columns_to_plot = ['mAP@0.5']

# Specify the path to your CSV files
csv_files = [
    "runs/[sparse_seunet]/[brightfield]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=False]-[losses=['labels', 'masks']]/[job=49178216]-[2023-11-12 15:57:38]/results.csv", 
    "runs/[sparse_seunet]/[brightfield]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=49177916]-[2023-11-12 14:19:32]/results.csv"
    ]

plot_from_csv(csv_files, columns_to_plot, interval=10)
