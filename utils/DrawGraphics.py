import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

# Data
methods = ['Ours', 'pix2pixHD', 'U-Net', 'ResNet', 'pix2pix_UNet', 'pix2pix_ResNet']
cd3_values = [0.16, 0.39, 1.00, 1.00, 0.36, 0.35]
panck_values = [0.07, 0.10, 0.12, 0.13, 0.11, 0.12]

# Colors (matching the provided figure)
colors = ['#7976A3', '#4B5F66', '#E39A56', '#BA5B59', '#4191C7', '#87B6A2']

# Bar width and positions
bar_width = 1
x_cd3 = np.arange(len(methods))
x_panck = x_cd3 + len(methods) + 0.5

# Plotting
fig, ax = plt.subplots(figsize=(8, 4), dpi=300)

# Bars for CD3
bars_cd3 = [ax.bar(x, cd3_values[i], bar_width, color=colors[i], edgecolor='black') for i, x in enumerate(x_cd3)]

# Bars for panCK
bars_panck = [ax.bar(x, panck_values[i], bar_width, color=colors[i], edgecolor='black') for i, x in enumerate(x_panck)]

# Adding text on top of bars
for bar in bars_cd3 + bars_panck:
    yval = bar[0].get_height()
    ax.text(bar[0].get_x() + bar[0].get_width() / 2, yval, round(yval, 2), ha='center', va='bottom')

# Labels and title
ax.set_xlabel('Stain')
ax.set_ylabel('MAE Ratio')
ax.set_title('Mean Absolute Error Ratios for CD3 and PanCK Positive Cells')
ax.set_xticks([x_cd3.mean(), x_panck.mean()])
ax.set_xticklabels(['CD3', 'panCK'])

# Custom legend
legend_handles = [Patch(facecolor=color, edgecolor='black', linewidth=1) for color in colors]
legend = ax.legend(legend_handles, methods, title='Method', loc='upper right', facecolor='lightgrey', edgecolor='black')  # Added facecolor for legend box


# Save plot
plt.savefig(r'D:\Chang_files\work_records\swinT\downstream\plot.png')

# Close the plot to avoid display
plt.close(fig)
