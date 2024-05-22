import os

import matplotlib
import pandas as pd
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import seaborn as sns
import re

# %matplotlib inline


# Mapping of the activities and colours
activity_codes_mapping = {'A': 'walking',
                          'B': 'jogging',
                          'C': 'stairs',
                          'D': 'sitting',
                          'E': 'standing',
                          'F': 'typing',
                          'G': 'brushing teeth',
                          'H': 'eating soup',
                          'I': 'eating chips',
                          'J': 'eating pasta',
                          'K': 'drinking from cup',
                          'L': 'eating sandwich',
                          'M': 'kicking soccer ball',
                          'O': 'playing catch tennis ball',
                          'P': 'dribbling basket ball',
                          'Q': 'writing',
                          'R': 'clapping',
                          'S': 'folding clothes'}

activity_color_map = {activity_codes_mapping['A']: 'lime',
                      activity_codes_mapping['B']: 'red',
                      activity_codes_mapping['C']: 'blue',
                      activity_codes_mapping['D']: 'orange',
                      activity_codes_mapping['E']: 'yellow',
                      activity_codes_mapping['F']: 'lightgreen',
                      activity_codes_mapping['G']: 'greenyellow',
                      activity_codes_mapping['H']: 'magenta',
                      activity_codes_mapping['I']: 'gold',
                      activity_codes_mapping['J']: 'cyan',
                      activity_codes_mapping['K']: 'purple',
                      activity_codes_mapping['L']: 'lightgreen',
                      activity_codes_mapping['M']: 'violet',
                      activity_codes_mapping['O']: 'limegreen',
                      activity_codes_mapping['P']: 'deepskyblue',
                      activity_codes_mapping['Q']: 'mediumspringgreen',
                      activity_codes_mapping['R']: 'plum',
                      activity_codes_mapping['S']: 'olive'}


def merge_text_files(directory, output_file_accel):
    with open(output_file_accel, 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join([open(os.path.join(directory, filename), 'r', encoding='utf-8').read()
                                 for filename in os.listdir(directory) if filename.endswith('.txt')]))


# Specify the directory and desired output file
directory = 'wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw/phone/accel'
output_file_accel = 'merged_text.txt'

# Call the function to merge the files
merge_text_files(directory, output_file_accel)

columns = ['subject_id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis']

df = pd.read_csv(output_file_accel, header=None, names=columns)

# Display the DataFrame
df = df.dropna()
df.head()

# Replace activity letter representation with activity name
df['activity'] = df['activity'].map(activity_codes_mapping)

# Display the DataFrame
df = df.dropna()
print(df.head())

# Checking for missing value in the data
missing_values = df.isnull().sum()
print("Missing values:")
print(missing_values)

df.info()

# Convert the z-axis to a float type
df['z-axis'] = df['z-axis'].map(lambda x: str(re.findall("\d+\.\d+", str(x))))
df['z-axis'] = df['z-axis'].map(lambda x: x[2:-2])
df['z-axis'] = pd.to_numeric(df['z-axis'], errors='coerce')

# Checking the information again to varify the conversion of the z-axis to float worked
df.info()

# Plotting a Histogram for the count of each activity

# Get the counts of each activity
activity_counts = df['activity'].value_counts()

# Get the corresponding colors from the activity_color_map
colors = [activity_color_map[activity] for activity in activity_counts.index]

# Specifying plot size
plt.figure(figsize=(14, 10))

# Plot the bar plot with specified colors and grid
activity_counts.plot(kind='bar', title='Histogram of Activity Counts', color=colors)
plt.grid(True)

# Add labels for better readability
plt.xlabel('Activity')
plt.ylabel('Count')

# Show plot
plt.show()

#
# def load_data_from_dir(directory, delimiter='\t'):
#     all_data = []
#     for filename in os.listdir(directory):
#         file_path = os.path.join(directory, filename)
#         if os.path.isfile(file_path) and filename.endswith('.txt'):
#             data = pd.read_csv(file_path, delimiter=delimiter)
#             all_data.append(data)
#         elif os.path.isdir(file_path):
#             # Recursively call the function for subdirectories
#             sub_data = load_data_from_dir(file_path, delimiter)
#             all_data.extend(sub_data)  # Extend the list with subdirectory data
#     return all_data
#
#
# # Define the main data directory (replace with your actual path)
# data_dir = 'wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw'
#
# # Load data from all subdirectories
# all_data = load_data_from_dir(data_dir)
#
# # Combine all DataFrames into a single one (optional)
# if all_data:  # Check if any files found
#     combined_data = pd.concat(all_data, ignore_index=True)
#     # Now you can work with the combined_data DataFrame
#     print(combined_data.head())  # View the first few rows
# else:
#     print("No TXT files found in the directory!")
#
#     # Choose two features for the scatter plot (e.g., x and y-axis accelerometer readings)
#     feature1 = 'x_axis_mean'
#     feature2 = 'y_axis_std'
#
#
# # Scatter plot of activities (assuming 'combined_data' contains all data)
#
# # Choose two features for the scatter plot (e.g., x and y-axis accelerometer readings)
# feature1 = 'x_axis_mean'
# feature2 = 'y_axis_std'
#
# # Create a scatter plot with different colors for each activity
# plt.figure(figsize=(10, 6))  # Adjust figure size as needed
# for activity in combined_data['activity'].unique():
#   activity_data = combined_data[combined_data['activity'] == activity]
#   plt.scatter(activity_data[feature1], activity_data[feature2], label=activity)
#
# # Add labels and title
# plt.xlabel(feature1)
# plt.ylabel(feature2)
# plt.title('Scatter plot of activities (color-coded)')
# plt.legend()  # Show legend for activities
#
# # Optional: Customize plot appearance (grid, markers, etc.)
# plt.grid(True)
#
# plt.show()

# def plot_subject_id(df):
#     '''Plots acceleration time history for each subject ID'''
#
#     # Get unique subject IDs
#     subjects = df['subject_id'].unique()
#
#     for subject_id in subjects:
#         data = df[df['subject_id'] == subject_id][['x-axis', 'y-axis', 'z-axis']][:200]
#         fig, ax = plt.subplots(figsize=(6, 3))  # Set the size of the plot
#
#         data["x-axis"].plot(color="b", label='x-axis')
#         data["y-axis"].plot(color="r", label='y-axis')
#         data["z-axis"].plot(color="g", label='z-axis')
#
#         # Add grid lines
#         ax.grid(True)
#         # Add legend
#         ax.legend()
#
#         plt.title('Acceleration: Subject ID - ' + str(subject_id))  # Set title
#         plt.xlabel('Time (index)')  # Set x-axis label
#         plt.ylabel('Acceleration (m/sec^2)')  # Set y-axis label
#
#         plt.tight_layout()  # Adjust layout to prevent overlap
#         plt.show()
#
# # Example usage of the function
# plot_subject_id(df)

# def plot_boxplot(df):
#     '''Plots a boxplot based on subject_id'''
#
#     plt.figure(figsize=(12, 10))
#     plt.boxplot([df[df['subject_id'] == subject]['x-axis'] for subject in df['subject_id'].unique()],
#                 labels=df['subject_id'].unique())
#     plt.title('Boxplot of x-axis Acceleration by Subject ID')
#     plt.xlabel('Subject ID')
#     plt.ylabel('Acceleration (m/sec^2)')
#
#      # Rotate x-axis labels vertically
#     plt.xticks(rotation=90)
#     # plt.grid(True)
#     plt.show()
#
# # Example usage
# plot_boxplot(df)


# def plot_boxplot(df):
#     '''Plots boxplot based on activity'''
#
#     plt.figure(figsize=(15, 3))
#     sns.boxplot(x='activity', y='x-axis', data=df)
#     plt.title('Boxplot of x-axis Acceleration by Activity')
#     plt.xlabel('Activity')
#     plt.ylabel('Acceleration (m/sec^2)')
#     plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
#     plt.grid(True)
#     plt.show()
#
# # Example usage of the function
# plot_boxplot(df)


# def plot_scatter_subject_id(df):
#     '''Plots scatter plot based on subject_id'''
#
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(data=df, x='subject_id', y='x-axis', hue='activity', palette='viridis')
#     plt.title('Scatter Plot of x-axis Acceleration by Subject ID')
#     plt.xlabel('Subject ID')
#     plt.ylabel('Acceleration (m/sec^2)')
#     plt.grid(True)
#     plt.legend(title='Activity', bbox_to_anchor=(1.05, 1), loc='upper left')
#     plt.tight_layout()
#     plt.show()
#
# # Example usage of the function
# plot_scatter_subject_id(df)


# def plot_scatter(df):
#     '''Plots scatter plots based on activity'''
#
#     # Define the pairs of axes for scatter plots
#     axes_pairs = [('x-axis', 'y-axis'), ('x-axis', 'z-axis'), ('y-axis', 'z-axis')]
#
#     # Create subplots
#     fig, axes = plt.subplots(nrows=1, ncols=len(axes_pairs), figsize=(15, 5))
#
#     # Plot scatter plots for each pair of axes
#     for idx, (x_axis, y_axis) in enumerate(axes_pairs):
#         sns.scatterplot(x=x_axis, y=y_axis, hue='activity', data=df, ax=axes[idx])
#         axes[idx].set_title(f'Scatter Plot of {x_axis} vs {y_axis}')
#         axes[idx].set_xlabel(x_axis)
#         axes[idx].set_ylabel(y_axis)
#         axes[idx].grid(True)
#
#     plt.tight_layout()
#     plt.show()
#
# # Example usage of the function
# plot_scatter(df)