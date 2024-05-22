import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set the directory paths
phone_accel_dir = 'wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw/phone/accel'
phone_gyro_dir = 'wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw/phone/gyro'
watch_accel_dir = 'wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw/watch/accel'
watch_gyro_dir = 'wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/raw/watch/gyro'


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


def read_data(directory):
    files = os.listdir(directory)
    dataframes = []
    for file in files:
        if file.endswith('.txt'):
            filepath = os.path.join(directory, file)
            df = pd.read_csv(filepath, header=None,
                             names=['subject_id', 'activity', 'timestamp', 'x-axis', 'y-axis', 'z-axis'])
            dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


phone_accel_data = read_data(phone_accel_dir)
phone_gyro_data = read_data(phone_gyro_dir)
watch_accel_data = read_data(watch_accel_dir)
watch_gyro_data = read_data(watch_gyro_dir)

phone_accel_data['device'] = 'phone'
phone_accel_data['sensor'] = 'accelerometer'

phone_gyro_data['device'] = 'phone'
phone_gyro_data['sensor'] = 'gyroscope'

watch_accel_data['device'] = 'watch'
watch_accel_data['sensor'] = 'accelerometer'

watch_gyro_data['device'] = 'watch'
watch_gyro_data['sensor'] = 'gyroscope'

merged_data = pd.concat([phone_accel_data, phone_gyro_data, watch_accel_data, watch_gyro_data], axis=0,
                        ignore_index=True)

# Check shape
merged_shape = merged_data.shape
total_shape = (
    phone_accel_data.shape[0] + phone_gyro_data.shape[0] + watch_accel_data.shape[0] + watch_gyro_data.shape[0],
    merged_data.shape[1])

print("Merged DataFrame Shape:", merged_shape)
print("Total Expected Shape:", total_shape)

# Replace activity letter representation with activity name
merged_data['activity'] = merged_data['activity'].map(activity_codes_mapping)
#
#
merged_data['z-axis'] = merged_data['z-axis'].map(lambda x: str(re.findall(r"\d+\.\d+", str(x))))
merged_data['z-axis'] = merged_data['z-axis'].map(lambda x: x[2:-2])
merged_data['z-axis'] = pd.to_numeric(merged_data['z-axis'], errors='coerce')

# Display the DataFrame
merged_data = merged_data.dropna()
print(merged_data.head())

# Checking for missing value in the data
missing_values = merged_data.isnull().sum()
print("Missing values:")
print(missing_values)

# # Get the counts of each activity
activity_counts = merged_data['activity'].value_counts()

# # Get the corresponding colors from the activity_color_map
colors = [activity_color_map[activity] for activity in activity_counts.index]

# 1. Histogram of Activity and Count

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

# # 2. Plotting a Histogram for the distribution of subject_id with grid and custom color

# Get the counts of each subject_id
subject_counts = merged_data['subject_id'].value_counts()

# Specifying plot size
plt.figure(figsize=(18, 10))

colors = [activity_color_map[activity] for activity in activity_counts.index]

# Plot the bar plot with grid and custom color
subject_counts.plot(kind='bar', title='Histogram of Phone Accel Subject IDs', color=colors)
plt.grid(True)  # Adding gridlines

# Add labels for better readability
plt.xlabel('Subject ID')
plt.ylabel('Count')

# Show plot
plt.show()

# # Plot histograms for each numerical column
for column in merged_data.select_dtypes(include=['float64', 'int64']).columns:
    if column == 'z-axis':
        data = merged_data[column].dropna()  # Drop NaN values for the 'z-axis'
    else:
        data = merged_data[column]  # For other columns, do not drop NaN values

    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=20, color='skyblue', edgecolor='black')
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


def plot_activities(df):
    # Get unique activity names
    derived_activities = df['activity'].unique()

    for activity in derived_activities:
        yy_data = df[df['activity'] == activity][['x-axis', 'y-axis', 'z-axis']][:200]
        fig, ax = plt.subplots(figsize=(6, 3))  # Set the size of the plot

        yy_data["x-axis"].plot(title=activity, color="b", label='x-axis')
        yy_data["y-axis"].plot(color="r", label='y-axis')
        yy_data["z-axis"].plot(color="g", label='z-axis')

        # Add grid lines
        ax.grid(True)
        # Add legend
        ax.legend()

        plt.title('Acceleration: Activity - ' + activity)  # Set title
        plt.xlabel('Time (sec)')  # Set x-axis label
        plt.ylabel('Acceleration (m/sec^2)')  # Set y-axis label

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()


plot_activities(merged_data)


def plot_subject_id(df):
    # Get unique subject IDs
    subjects = df['subject_id'].unique()

    for derived_subject_id in subjects:
        xx_data = df[df['subject_id'] == derived_subject_id][['x-axis', 'y-axis', 'z-axis']][:200]
        fig, ax = plt.subplots(figsize=(6, 3))  # Set the size of the plot

        xx_data["x-axis"].plot(color="b", label='x-axis')
        xx_data["y-axis"].plot(color="r", label='y-axis')
        xx_data["z-axis"].plot(color="g", label='z-axis')

        # Add grid lines
        ax.grid(True)
        # Add legend
        ax.legend()

        plt.title('Acceleration: Subject ID - ' + str(derived_subject_id))  # Set title
        plt.xlabel('Time (index)')  # Set x-axis label
        plt.ylabel('Acceleration (m/sec^2)')  # Set y-axis label

        plt.tight_layout()  # Adjust layout to prevent overlap
        plt.show()


plot_subject_id(merged_data)


def plot_boxplot(df):
    plt.figure(figsize=(12, 10))
    plt.boxplot([df[df['subject_id'] == subject]['x-axis'] for subject in df['subject_id'].unique()],
                labels=df['subject_id'].unique())
    plt.title('Boxplot of x-axis by Subject ID')
    plt.xlabel('Subject ID')
    plt.ylabel('X-Axis')

    # Rotate x-axis labels vertically
    plt.xticks(rotation=90)
    # plt.grid(True)
    plt.show()


def plot_activities(df):
    # Get unique activity names
    unique_activities = df['activity'].unique()

    for activity in unique_activities:
        # Filter the DataFrame based on the activity
        activity_data = df[df['activity'] == activity][:200]

        # Create subplots
        fig, ax = plt.subplots(figsize=(6, 3))  # Set the size of the plot

        # Plot each axis
        activity_data["x-axis"].plot(title=activity, color="b", label='x-axis')
        activity_data["y-axis"].plot(color="r", label='y-axis')
        activity_data["z-axis"].plot(color="g", label='z-axis')

        # Add grid lines and legend
        ax.grid(True)
        ax.legend()

        # Set title and axis labels
        plt.title('Gyroscope: Activity - ' + activity)
        plt.xlabel('Time (sec)')
        plt.ylabel('Gyroscope (radius/sec)')

        plt.tight_layout()
        plt.show()


plot_activities(merged_data)

# Plot pie chart for activities
plt.figure(figsize=(8, 8))
merged_data['activity'].value_counts().plot(kind='pie', autopct='%1.1f%%',
                                            colors=[activity_color_map[a] for a in merged_data['activity'].unique()])
plt.title('Pie Chart of Activities')
plt.ylabel('')
plt.show()

# Plot pie chart for devices
plt.figure(figsize=(8, 8))
merged_data['device'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['lightblue', 'lightgreen'])
plt.title('Pie Chart of Devices')
plt.ylabel('')
plt.show()

# Plot pie chart for subject IDs
plt.figure(figsize=(8, 8))
merged_data['subject_id'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Pie Chart of Subject IDs')
plt.ylabel('')
plt.show()
#

# Set the style of seaborn
sns.set(style="whitegrid")

# Create a boxplot for all activities
plt.figure(figsize=(12, 10))
sns.boxplot(x='activity', y='x-axis', data=merged_data, palette=activity_color_map)
plt.title('Boxplot for All Activities')
plt.xlabel('Activity Code')
plt.ylabel('Acceleration (m/sec^2)')
plt.xticks(rotation=45)
plt.show()

# Define a list of activities
activities = merged_data['activity'].unique()

# Plot scatter plots for individual activities
for activity in activities:
    activity_data = merged_data[merged_data['activity'] == activity]
plt.figure(figsize=(8, 6))
plt.scatter(activity_data['x-axis'], activity_data['y-axis'], label=activity, color=activity_color_map[activity])
plt.title(f'Scatter Plot for {activity}')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.grid(True)
plt.show()

# Plot scatter plot comparing all activities with other variables
plt.figure(figsize=(10, 8))
for activity in activities:
    activity_data = merged_data[merged_data['activity'] == activity]
    plt.scatter(activity_data['x-axis'], activity_data['y-axis'], label=activity,
                color=activity_color_map[activity], alpha=0.5)
plt.title('Scatter Plot of All Activities')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend(title='Activity')
plt.grid(True)
plt.show()

# Plot scatter plot for devices
plt.figure(figsize=(8, 6))
for device in merged_data['device'].unique():
    device_data = merged_data[merged_data['device'] == device]
    plt.scatter(device_data.index, device_data['activity'], label=device)
plt.title('Scatter Plot of Activities by Device')
plt.xlabel('Index')
plt.ylabel('Activity')
plt.legend(title='Device')
plt.grid(True)
plt.show()

# Plot scatter plot for sensors
plt.figure(figsize=(8, 6))
for sensor in merged_data['sensor'].unique():
    sensor_data = merged_data[merged_data['sensor'] == sensor]
    plt.scatter(sensor_data.index, sensor_data['activity'], label=sensor)
plt.title('Scatter Plot of Activities by Sensor')
plt.xlabel('Index')
plt.ylabel('Activity')
plt.legend(title='Sensor')
plt.grid(True)
plt.show()
#

# Plot scatter plot comparing timestamp with activities
plt.figure(figsize=(10, 8))
for activity in merged_data['activity'].unique():
    activity_data = merged_data[merged_data['activity'] == activity]
    plt.scatter(activity_data['timestamp'], activity_data.index, label=activity, color=activity_color_map[activity],
                alpha=0.5)
plt.title('Scatter Plot of Timestamp with Activities')
plt.xlabel('Timestamp')
plt.ylabel('Index')
plt.legend(title='Activity')
plt.grid(True)
plt.show()

# Create a dataframe for devices
device_data = merged_data.pivot_table(index='device', columns='activity', aggfunc='size', fill_value=0)

# Create a heatmap for devices
plt.figure(figsize=(10, 8))
sns.heatmap(device_data, cmap='viridis', annot=False, fmt='d')
plt.title('Heatmap of Activities by Device')
plt.xlabel('Activity')
plt.ylabel('Device')
plt.show()

# Create a dataframe for sensors
sensor_data = merged_data.pivot_table(index='sensor', columns='activity', aggfunc='size', fill_value=0)

# Create a heatmap for sensors
plt.figure(figsize=(10, 8))
sns.heatmap(sensor_data, cmap='viridis', annot=True, fmt='d')
plt.title('Heatmap of Activities by Sensor')
plt.xlabel('Activity')
plt.ylabel('Sensor')
plt.show()
#
#
# Create a heatmap comparing timestamp with activities
timestamp_data = merged_data.pivot_table(index='timestamp', columns='activity', aggfunc='size', fill_value=0)

# Create a heatmap for timestamp with activities
plt.figure(figsize=(12, 8))
sns.heatmap(timestamp_data, cmap='viridis', annot=False, fmt='d')
plt.title('Heatmap of Timestamp with Activities')
plt.xlabel('Activity')
plt.ylabel('Timestamp')
plt.show()

# plt.figure(figsize=(12, 6))
for subject_id, subject_data in merged_data.groupby('subject_id'):
    plt.plot(subject_data['timestamp'], subject_data['activity'], label=f"Subject ID {subject_id}")
plt.title('Time Series Plot of Activities by Timestamp and Subject IDs')
plt.xlabel('Timestamp')
plt.ylabel('Activity')
plt.xticks(rotation=45)
plt.legend(title='Subject IDs')
plt.grid(True)
plt.show()

# Filter accelerometer data for phones and watches
phone_accel_data = merged_data[(merged_data['device'] == 'phone') & (merged_data['sensor'] == 'accelerometer')]
watch_accel_data = merged_data[(merged_data['device'] == 'watch') & (merged_data['sensor'] == 'accelerometer')]

# Plot sensor data
plt.figure(figsize=(12, 6))
sns.lineplot(x='timestamp', y='x-axis', data=phone_accel_data, label='Phone Accelerometer')
sns.lineplot(x='timestamp', y='x-axis', data=watch_accel_data, label='Watch Accelerometer')
plt.title('Comparison of Accelerometer Data (X-axis) between Phone and Watch')
plt.xlabel('Timestamp')
plt.ylabel('Acceleration (m/sec^2)')
plt.legend()
plt.grid(True)
plt.show()

# Sample a subset of the data with a million records
subset_data = merged_data.sample(n=10 ** 6, random_state=42)

# Drop rows with missing values
subset_data = subset_data.dropna()

# Convert timestamp to datetime
subset_data['timestamp'] = pd.to_datetime(subset_data['timestamp'], unit='ms', errors='coerce')

# Select relevant features
features = ['x-axis', 'y-axis', 'z-axis']  # Adjust the features as needed

# Split the data into training and testing sets
X_subset = subset_data[features]
y_subset = subset_data['activity']
X_train_subset, X_test_subset, y_train_subset, y_test_subset = train_test_split(X_subset, y_subset, test_size=0.2,
                                                                                random_state=42)

# Create and train the Random Forest model on the subset
model_subset = RandomForestClassifier(n_estimators=100, random_state=42)
model_subset.fit(X_train_subset, y_train_subset)

# Evaluate the model on the testing data
y_test_pred_subset = model_subset.predict(X_test_subset)
test_accuracy_subset = accuracy_score(y_test_subset, y_test_pred_subset)
test_confusion_mat_subset = confusion_matrix(y_test_subset, y_test_pred_subset)
test_classification_rep_subset = classification_report(y_test_subset, y_test_pred_subset)

print("Testing Accuracy on Subset:", test_accuracy_subset)
print("Testing Confusion Matrix on Subset:")
print(test_confusion_mat_subset)
print("Testing Classification Report on Subset:")
print(test_classification_rep_subset)
