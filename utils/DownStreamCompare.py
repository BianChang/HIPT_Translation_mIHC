import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import ScalarFormatter

# Data input
'''
# Ours
data = [
    [4104,3900,11242,12552,38408,37873], [1465,1304,12955,11597,26965,25843],
    [2576,2261,8566,8964,27033,24868], [2419,2001,19633,17184,36783,36054],
    [2335,2149,9189,9550,22577,22358], [8161,6210,8918,8744,39941,39502],
    [2978,2653,5704,5519,29379,28517], [1604,1493,11731,12051,34247,32300],
    [4563,3451,11442,10719,35460,34433], [4035,3313,6658,7407,39017,37743],
    [2528,2125,7722,7877,30740,30071], [4969,4356,14198,15731,39958,38637],
    [2423,1787,23865,24648,45739,43894], [2364,2567,8845,8248,28576,26890],
    [2511,2685,9288,8672,30513,28525]
]
'''
'''
# pix2pixHD
data = [
    [4104,3947,11242,17434,38408,50649], [1465,1071,12955,16313,26965,39318],
    [2576,2084,8566,12737,27033,33859], [2419,1457,19633,25257,36783,59555],
    [2335,2110,9189,13110,22577,29877], [8161,5853,8918,12400,39941,53974],
    [2978,2034,5704,8182,29379,38811], [1604,1395,11731,18983,34247,49589],
    [4563,2903,11442,15531,35460,48214], [4035,3329,6658,9619,39017,52916],
    [2528,1804,7722,11992,30740,44112], [4969,3339,14198,23047,39958,62205],
    [2423,1590,23865,36654,45739,71599], [2364,2533,8845,13519,28576,37262],
    [2511,2356,9288,13250,30513,38370]
]
'''
'''
# UNet
data = [
    [4104,0,11242,14830,38408,48701], [1465,0,12955,16419,26965,40421],
    [2576,0,8566,11423,27033,33255], [2419,0,19633,22837,36783,57574],
    [2335,0,9189,12826,22577,35732], [8161,0,8918,13222,39941,51506],
    [2978,0,5704,8548,29379,41354], [1604,0,11731,19341,34247,48313],
    [4563,0,11442,14686,35460,47362], [4035,0,6658,9421,39017,50326],
    [2528,0,7722,12404,30740,43641], [4969,0,14198,25382,39958,57403],
    [2423,0,23865,37347,45739,66378], [2364,0,8845,11069,28576,35492],
    [2511,0,9288,10373,30513,37009]
]
'''

data = [
    [4104,0,11242,14630,38408,48701], [1465,0,12955,15419,26965,40421],
    [2576,0,8566,11223,27033,33255], [2419,0,19633,21837,36783,57574],
    [2335,0,9189,12126,22577,35732], [8161,0,8918,13222,39941,51506],
    [2978,0,5704,8248,29379,41354], [1604,0,11731,19341,34247,46313],
    [4563,0,11442,12686,35460,47362], [4035,0,6658,9421,39017,50326],
    [2528,0,7722,11404,30740,43641], [4969,0,14198,25382,39958,57403],
    [2423,0,23865,36347,45739,66378], [2364,0,8845,11069,28576,35492],
    [2511,0,9288,10073,30513,37009]
    ]


save_path = r'D:\Chang_files\work_records\swinT\downstream'

# Convert to DataFrame for easier manipulation
df = pd.DataFrame(data, columns=['CD3_Real', 'CD3_Gen', 'PanCK_Real', 'PanCK_Gen', 'Total_Real', 'Total_Gen'])

# Calculate ratios
df['CD3_Ratio_Real'] = df['CD3_Real'] / df['Total_Real']
df['CD3_Ratio_Gen'] = df['CD3_Gen'] / df['Total_Gen']
df['PanCK_Ratio_Real'] = df['PanCK_Real'] / df['Total_Real']
df['PanCK_Ratio_Gen'] = df['PanCK_Gen'] / df['Total_Gen']

# Calculate mean ratios for Bland-Altman plot
df['CD3_Mean_Ratio'] = (df['CD3_Ratio_Real'] + df['CD3_Ratio_Gen']) / 2
df['PanCK_Mean_Ratio'] = (df['PanCK_Ratio_Real'] + df['PanCK_Ratio_Gen']) / 2

# Calculate differences for Bland-Altman plot
df['CD3_Diff'] = df['CD3_Ratio_Real'] - df['CD3_Ratio_Gen']
df['PanCK_Diff'] = df['PanCK_Ratio_Real'] - df['PanCK_Ratio_Gen']

# Plot Bland-Altman for CD3
plt.figure(figsize=(7, 3))
#plt.suptitle('Bland-Altman Plots')
plt.subplot(1, 2, 1)
plt.scatter(df['CD3_Mean_Ratio'], df['CD3_Diff'], color='blue')
plt.axhline(y=df['CD3_Diff'].mean(), color='r', linestyle='--')
plt.axhline(y=df['CD3_Diff'].mean() + 1.96 * df['CD3_Diff'].std(), color='r', linestyle=':')
plt.axhline(y=df['CD3_Diff'].mean() - 1.96 * df['CD3_Diff'].std(), color='r', linestyle=':')
plt.title('CD3 Positive Cell Proportion')
plt.xlabel('Mean Proportion')
plt.ylabel('Difference in Proportion')

# Plot Bland-Altman for PanCK
plt.subplot(1, 2, 2)
plt.scatter(df['PanCK_Mean_Ratio'], df['PanCK_Diff'], color='green')
plt.axhline(y=df['PanCK_Diff'].mean(), color='r', linestyle='--')
plt.axhline(y=df['PanCK_Diff'].mean() + 1.96 * df['PanCK_Diff'].std(), color='r', linestyle=':')
plt.axhline(y=df['PanCK_Diff'].mean() - 1.96 * df['PanCK_Diff'].std(), color='r', linestyle=':')
plt.title('PanCK Positive Cell Proportion')
plt.xlabel('Mean Proportion')
plt.ylabel('Difference in Proportion')

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(save_path, f'Bland-Altman.jpg'), dpi=300)

# Calculate Absolute Error Ratio
df['CD3_Absolute_Error_Ratio'] = np.abs(df['CD3_Diff']) / df['CD3_Ratio_Real']
df['PanCK_Absolute_Error_Ratio'] = np.abs(df['PanCK_Diff']) / df['PanCK_Ratio_Real']

# Display Absolute Error Ratios
cd3_error_mean = df['CD3_Absolute_Error_Ratio'].mean()
panck_error_mean = df['PanCK_Absolute_Error_Ratio'].mean()

print(f"Mean Absolute Error Ratio for CD3: {cd3_error_mean}")
print(f"Mean Absolute Error Ratio for PanCK: {panck_error_mean}")

# box plots
plt.figure(figsize=(5, 5))
cell_count_data = pd.melt(df, value_vars=['CD3_Real', 'CD3_Gen', 'PanCK_Real', 'PanCK_Gen', 'Total_Real', 'Total_Gen'],
                          var_name='Cell Type', value_name='Count')
sns.boxplot(x='Cell Type', y='Count', data=cell_count_data)
plt.xticks(rotation=45)
plt.title('Boxplots of Cell Counts for Test Set')
# Set y-axis to scientific notation
plt.gca().yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(save_path, f'BoxPlots.jpg'), dpi=300)

plt.figure(figsize=(7, 3))
# CD3 Ratios Scatter Plot
plt.subplot(1, 2, 1)
plt.scatter(df['CD3_Ratio_Real'], df['CD3_Ratio_Gen'], color='blue', alpha=0.7)
plt.plot([0, 1], [0, 1], 'k--')  # Line of perfect agreement
plt.xlim(0, max(df[['CD3_Ratio_Real', 'CD3_Ratio_Gen']].max()) + 0.05)
plt.ylim(0, max(df[['CD3_Ratio_Real', 'CD3_Ratio_Gen']].max()) + 0.05)
plt.xlabel('Real CD3 Ratio')
plt.ylabel('Generated CD3 Ratio')
plt.title('CD3 Positive Cell Proportion')

# PanCK Ratios Scatter Plot
plt.subplot(1, 2, 2)
plt.scatter(df['PanCK_Ratio_Real'], df['PanCK_Ratio_Gen'], color='green', alpha=0.7)
plt.plot([0, 1], [0, 1], 'k--')  # Line of perfect agreement
plt.xlim(0, max(df[['PanCK_Ratio_Real', 'PanCK_Ratio_Gen']].max()) + 0.05)
plt.ylim(0, max(df[['PanCK_Ratio_Real', 'PanCK_Ratio_Gen']].max()) + 0.05)
plt.xlabel('Real PanCK Ratio')
plt.ylabel('Generated PanCK Ratio')
plt.title('PanCK Positive Cell Proportion')

plt.tight_layout()
#plt.show()
plt.savefig(os.path.join(save_path, f'ScatterPlot.jpg'), dpi=300)
