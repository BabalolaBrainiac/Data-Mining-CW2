# README for WISDM Dataset Analysis

## Introduction
This repository contains code for analyzing the WISDM Smartphone and Smartwatch Activity and Biometrics Dataset. The analysis involves data loading, preprocessing, visualization, and building a Random Forest classification model to predict activities based on sensor data, and was done in partial fulfilment of a Msc in Computing as part of my course works,

## Dataset
The WISDM dataset includes accelerometer and gyroscope data from smartphones and smartwatches worn by 51 subjects performing 18 different activities.

## Directory Structure
- `wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/`
  - `wisdm-dataset/`
    - `raw/`
      - `phone/`
        - `accel/` - Raw accelerometer data from the phone
        - `gyro/` - Raw gyroscope data from the phone
      - `watch/`
        - `accel/` - Raw accelerometer data from the watch
        - `gyro/` - Raw gyroscope data from the watch

## Code Overview

### Libraries
The following libraries are used in the analysis:
- `os`
- `pandas`
- `matplotlib.pyplot`
- `seaborn`
- `re`
- `sklearn`

### Data Reading and Preprocessing
1. **Set Directory Paths**: Define the paths for the accelerometer and gyroscope data for both phones and watches.
2. **Activity Mapping**: Map activity codes to activity names and assign colors for visualization.
3. **Read Data**: Function `read_data(directory)` reads and concatenates all sensor data files in the specified directory.
4. **Data Merging**: Combine data from all sensors into a single DataFrame and add `device` and `sensor` columns.
5. **Preprocessing**: Replace activity codes with activity names, clean the `z-axis` column, and handle missing values.

### Exploratory Data Analysis (EDA)
1. **Activity Counts**: Plot a histogram of activity counts.
2. **Subject Counts**: Plot a histogram of subject counts.
3. **Histograms for Each Numerical Column**: Plot histograms for `x-axis`, `y-axis`, and `z-axis`.
4. **Activity Time Series**: Plot time series of sensor data for each activity.
5. **Subject Time Series**: Plot time series of sensor data for each subject.
6. **Boxplots**: Plot boxplots for `x-axis` values grouped by subject IDs.
7. **Pie Charts**: Plot pie charts for activity distribution, device distribution, and subject distribution.
8. **Scatter Plots**: Plot scatter plots for sensor data by activities, devices, and sensors.
9. **Heatmaps**: Plot heatmaps for activity distribution by devices and sensors.

### Machine Learning Model
1. **Subset Data**: Sample a subset of the data with a million records.
2. **Feature Selection**: Select relevant features (`x-axis`, `y-axis`, `z-axis`).
3. **Train-Test Split**: Split the data into training and testing sets.
4. **Random Forest Model**: Train a Random Forest model on the subset of data.
5. **Model Evaluation**: Evaluate the model on the testing data using accuracy, confusion matrix, and classification report.

## How to Use

### Prerequisites
Ensure the following Python libraries are installed:
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

You can install them using pip:
```sh
pip install pandas matplotlib seaborn scikit-learn
```

### Running the Code
1. **Download the Dataset**: Ensure the WISDM dataset is downloaded and placed in the correct directory structure.
2. **Run the Script**: Execute the Python script to perform data loading, preprocessing, visualization, and model building.

```sh
python your_script_name.py
```

## Output
The script will output various plots for EDA and print the results of the Random Forest model evaluation, including accuracy, confusion matrix, and classification report.

## Contact
For any questions or further information, please contact Babalola Opeyemi Daniel at babalolaopedaniel@gmail.com.