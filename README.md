<p align="center">
<img width="495" height="182" alt="image" src="https://github.com/user-attachments/assets/52261f21-4cd9-49a3-9ce2-fa863d33bb22" />  

<p align="center">
DEPARTMENT OF ENGINEERING (ELECTRICAL-MECHATRONICS)
<p align="center">
FACULTY OF ELECTRICAL ENGINEERING 
<p align="center">
UNIVERSITI TEKNOLOGI MALAYSIA
<p align="center">


---

<p align="center">
SKEE1033 - 17 SCIENTIFIC PROGRAMMING
<p align="center">
ASSIGNMENT
  
---

<p align="center">
TITLE :ELECTRICITY CONSUMPTION AND COST ANALYSIS
<p align="center">
PREPARED BY : GROUP SHAOLIN MONKEY

<div align="center">
  
| NAME | MATRIC NO. |
| :--- | :----: |
| MUHAMAD DANISH BIN MOHAMAD NASIR | A25KE0277 |
| LEW ZHILY | A25KE5012 |
| CHUA JIAN BIN | A25KE0069  |
| MOHAMAD ZAKARIA BIN MISKARA | A25KE0275 |
| LEW YONG SHENG | A25KE5007 |

</div>

<p align="center">
PREPARED FOR: DR MUHAMMAD MUN'IM AHMAD ZABIDI


# INTRODUCTION

<p style="text-align: justify; text-indent: 5em;">

The SKEE 1033 Group Assignment is an extensive work on the topic of electricity consumption in households and the expenses involved in these regions, in Urban, Suburban and Rural areas. Based on a longitudinal dataset that covers the period between 2018 and 2022, this project will examine the complex correlation between the pattern of energy consumption and household size, in this case, the number of occupants. Since electricity prices are a major consideration in budgeting in the household, this paper will use data science methods (Python-based) such as data cleaning, statistical analysis, and visualization to extract insights into the historical data. Through the combination of these analytical tools, the project will offer a systematic way of exploring the impact of regional factors and occupancy on energy demand in a contemporary setting.

</p>

# OBJECTIVE

The main task of this assignment is to analyze and model the consumption of electricity in detail and predictively in order to estimate the costs of electricity spent by a family in a month. To do this, the project will apply a procedural cleaning of data including missing values and removal of the outliers through various Interquartile Range (IQR) to guarantee the data integrity. Moreover, the paper aims to identify the relationship between the number of people living in households and the level of consumption alongside representing regional tendencies with the help of multi-line and scatter graphs. Lastly, the project will design a Linear Regression model to predict the costs of electricity (Cost_RM) and use the predictive quality of the model based on certain performance measures like R-squared and Mean Absolute Error (MAE).


# Task 1: Data Cleaning Process

1. Inspect the dataset for missing values and outliers using Python libraries such as pandas and numpy.
2. Document the number of missing values and describe how you will handle them (e.g., using mean, median, or interpolation).
3. Detect and handle outliers in electricity consumption and cost using the Interquartile Range (IQR) or Z-score method.

Sample code:

<p align=center>

<img width="767" height="279" alt="image" src="https://github.com/user-attachments/assets/3317e3ff-4191-4e7a-9709-1bd7f131bda0" />

**Code:**

    from google.colab import files
    files.upload()

    import numpy as np
    import pandas as pd
    import os
    import operator as op

    # -----------------------------------------------------
    # Task 1: Data Cleaning Process
    # Identifying and handling missing values and outliers
    # -----------------------------------------------------

    # -----------------------------------------------------
    # 1. Load the dataset
    # -----------------------------------------------------
    file_path = '/content/dataset_student.csv'
    df = pd.read_csv(file_path)

    # -----------------------------------------------------
    # 2. Inspect for missing values
    # -----------------------------------------------------
    missing_values = df.isnull().any(axis=1)
    print('Missing Values:')
    print(df.loc[missing_values])

    # -----------------------------------------------------
    # 3. Handle missing values using interpolation
    # -----------------------------------------------------
    # Interpolate 'Consumption_kWh' and 'Cost_RM', then round to 2 decimal places.
    # Interpolate 'Occupants' and round to the nearest whole number.
    df['Consumption_kWh'] = df['Consumption_kWh'].interpolate().round(2)
    df['Cost_RM'] = df['Cost_RM'].interpolate().round(2)
    df['Occupants'] = df['Occupants'].interpolate().round(0)

    # Print rows that originally had missing values to show the effect of interpolation
    print('\nAfter Replacing Missing Values:')
    print(df.loc[missing_values])

    # -----------------------------------------------------
    # 4. Detect and handle outliers in 'Consumption_kWh'
    # -----------------------------------------------------
    # Calculate the Interquartile Range (IQR) for 'Consumption_kWh'
    Q1 = df['Consumption_kWh'].quantile(0.25)
    Q3 = df['Consumption_kWh'].quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier bounds (1.5 * IQR above Q3 and below Q1)
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify rows containing outliers in 'Consumption_kWh'
    df['Outlier'] = (df['Consumption_kWh'] < lower_bound) | (df['Consumption_kWh'] > upper_bound)

    # Print rows identified as outliers
    print('\nOutlier Values:')
    print(df.loc[df['Outlier']])

    # Replace outliers with the median 'Consumption_kWh' value
    median_value = round(df['Consumption_kWh'].median(), 2)
    df.loc[df['Outlier'], 'Consumption_kWh'] = median_value

    # -----------------------------------------------------
    # 5. Recalculate 'Cost_RM' based on cleaned 'Consumption_kWh'
    # -----------------------------------------------------
    # Recalculate 'Cost_RM' using a rate of 0.57 per kWh and round to 2 decimal places.

    df['Cost_RM'] = (df['Consumption_kWh'] * 0.57).round(2)

    # Print the outlier rows again to confirm replacement and cost recalculation
    print('\nOutliers After Replacement:')
    print(df.loc[df['Outlier']])

    # -----------------------------------------------------
    # 6. Save the cleaned DataFrame
    # -----------------------------------------------------
    df.to_csv('Cleaned_dataset_student.csv', index=False)

**Output:**

<img width="509" height="295" alt="image" src="https://github.com/user-attachments/assets/c09667c8-2e33-49b6-aa2a-921e7170e856" />

<img width="505" height="298" alt="image" src="https://github.com/user-attachments/assets/3fe1e0d7-9c86-44f3-a0ef-b9eed030aa13" />

<img width="562" height="163" alt="image" src="https://github.com/user-attachments/assets/aa305c3c-e48a-4eaa-bffd-f78be204b51f" />


# Task 2: Descriptive Data Analysis
1. Compute summary statistics (mean, median, and standard deviation) for Consumption_kWh and
Cost_RM by region.
2. Analyse the relationship between consumption and number of occupants using a correlation
coefficient.
3. Comment on any observable relationship patterns.

Sample code:

<p align=center>
<img width="769" height="113" alt="image" src="https://github.com/user-attachments/assets/6293f8c4-3366-41e5-a976-7d9b268823af" />

**Code:**

    import numpy as np
    import pandas as pd
    import os

    # -----------------------------------------------------
    # Task 2: Descriptive Analysis
    # Generating summary statistics and correlation analysis
    # -----------------------------------------------------

    # -----------------------------------------------------
    # 1. Load the cleaned dataset
    # -----------------------------------------------------
    df = pd.read_csv('Cleaned_dataset_student.csv')

    # -----------------------------------------------------
    # 2. Generate summary statistics by region
    # -----------------------------------------------------
    # Group data by 'Region' and calculate mean, median, and standard deviation
    # for 'Consumption_kWh' and 'Cost_RM', rounding results to 4 decimal places.
    summary = df.groupby('Region')[['Consumption_kWh', 'Cost_RM']].agg(
        ['mean', 'median', 'std']
        ).round(4)

    # Display the summary statistics table
    print('\nSummary Statistics by Region:')
    print(summary)

    # -----------------------------------------------------
    # 3. Calculate correlation between 'Consumption_kWh' and 'Occupants'
    # -----------------------------------------------------
    # Compute the correlation matrix for the specified columns
    correlation = df[['Consumption_kWh', 'Occupants']].corr()

    # Extract the specific correlation coefficient between 'Consumption_kWh' and 'Occupants'
    correlation_value = correlation.iloc[0, 1]

    # Display the full correlation matrix
    print('\nCorrelation Matrix:')
    print(correlation)

    # -----------------------------------------------------
    # 4. Interpret the correlation result
    # -----------------------------------------------------
    # Provide a comment based on whether the correlation is positive or negative
    if correlation_value > 0:
        print('Comment: Higher number of occupants is associated with higher electricity consumption.')
    else:
        print('Comment: There is a weak or negative relationship between occupants and electricity consumption.')

**Output:**

<img width="690" height="228" alt="image" src="https://github.com/user-attachments/assets/f3fecb57-17eb-43d4-81fe-7827048ac2cf" />


# Task 3: Data Visualisation
1. Plot monthly consumption trends for each region using line graphs.
2. Create a multi-line plot comparing all three regions.
3. Plot a scatter plot showing the relationship between the number of occupants and electricity
consumption.
4. Explain any outliers or unusual patterns observed in the graphs, such as sudden spikes in
consumption or irregular points in the scatter plot.

Sample code:

<p align=center>
<img width="768" height="328" alt="image" src="https://github.com/user-attachments/assets/8b4b0c28-7249-425a-b017-7b441029b67a" />


**Code:**

    import pandas as pd
    import matplotlib.pyplot as plt

    # -----------------------------------------------------
    # Task 3: Data Visualisation
    # Plotting trends and relationships in electricity data
    # -----------------------------------------------------

    # -----------------------------------------------------
    # 1. Load the cleaned dataset
    # -----------------------------------------------------
    df = pd.read_csv('Cleaned_dataset_student.csv')

    # -----------------------------------------------------
    # 2. Monthly consumption trends for each region (separated by year)
    # -----------------------------------------------------
    # Iterate through each unique region to create a dedicated plot.
    for region in df['Region'].unique():
        region_data = df[df['Region'] == region] # Filter data for the current region
        plt.figure(figsize=(10, 6)) # Create a new figure for each region's plot

        # Plot consumption trends for each year within the current region
        for year in sorted(region_data['Year'].unique()):
            # Filter data for the current year and sort by month for correct chronological plotting
            year_data = region_data[region_data['Year'] == year].sort_values('Month')
            plt.plot(
                year_data['Month'], # X-axis: Month (1-12)
                year_data['Consumption_kWh'], # Y-axis: Electricity Consumption in kWh
                marker='o', # Add circular markers to data points
                label=str(year) # Label each line with its corresponding year
            )

        plt.xticks(range(1, 13)) # Ensure X-axis ticks are displayed for all 12 months
        plt.xlabel('Month') # Label for the X-axis
        plt.ylabel('Electricity Consumption (kWh)') # Label for the Y-axis
        plt.title(f'Monthly Electricity Consumption Trend in {region} (2018-2022)') # Title of the plot
        plt.legend(title='Year') # Display legend showing years with the title 'Year'
        plt.grid(True) # Add a grid to the plot for enhanced readability
        plt.show() # Display the generated plot

    # -----------------------------------------------------
    # 3. Multi-line plot comparing all regions (average monthly consumption)
    # -----------------------------------------------------
    # Create a single figure to compare average monthly consumption across all regions
    plt.figure(figsize=(10, 6))

    # Calculate and plot the average monthly consumption for each region
    for region in df['Region'].unique():
        region_data = df[df['Region'] == region] # Filter data for the current region
        # Calculate the average 'Consumption_kWh' for each month across all years for this region
        monthly_avg = region_data.groupby('Month')['Consumption_kWh'].mean()
        plt.plot(monthly_avg.index, monthly_avg.values, marker='o', label=region) # Plot the average trend

    plt.xticks(range(1, 13)) # Ensure X-axis ticks are displayed for all 12 months
    plt.xlabel('Month') # Label for the X-axis
    plt.ylabel('Average Electricity Consumption (kWh)') # Label for the Y-axis
    plt.title('Average Monthly Electricity Consumption by Region (2018-2022)') # Title of the plot
    plt.legend() # Display legend showing region names
    plt.grid(True) # Add a grid to the plot
    plt.show() # Display the generated plot

    # -----------------------------------------------------
    # 4. Scatter plot: Occupants vs Electricity Consumption
    # -----------------------------------------------------
    # Create a single figure for the scatter plot to show the relationship between occupants and consumption
    plt.figure(figsize=(10, 6))
    # Plot 'Occupants' on the X-axis and 'Consumption_kWh' on the Y-axis
    plt.scatter(df['Occupants'], df['Consumption_kWh'], alpha=0.7) # Add transparency to overlapping points
    plt.xlabel('Number of Occupants') # Label for the X-axis
    plt.ylabel('Electricity Consumption (kWh)') # Label for the Y-axis
    plt.title('Relationship Between Number of Occupants and Electricity Consumption') # Title of the plot
    plt.grid(True) # Add a grid to the plot
    plt.show() # Display the generated plot

    # -----------------------------------------------------
    # 5. Box Plot: Electricity Consumption by Region
    # -----------------------------------------------------
    # Create a figure for the box plot
    plt.figure(figsize=(10, 6))

    # Create box plot comparing electricity consumption across regions
    df.boxplot(
        column='Consumption_kWh',  # Data to plot
        by=' Region'                # Group data by region
    )

    plt.xlabel('Region')  # X-axis label
    plt.ylabel('Electricity Consumption (kWh)')  # Y-axis label
    plt.title('Electricity Consumption Distribution by Region')  # Plot title
    plt.suptitle('')  # Remove default pandas subtitle
    plt.grid(True)  # Add grid for readability
    plt.show()  # Display the plot

**Output:**

<img width="850" height="547" alt="image" src="https://github.com/user-attachments/assets/fd2769b4-33b8-4a28-9c55-65301d78dd6e" />

<img width="850" height="547" alt="image" src="https://github.com/user-attachments/assets/c3b79e5d-2ab8-4508-8bce-f40f58a1a5cc" />

<img width="850" height="547" alt="image" src="https://github.com/user-attachments/assets/24d34d51-1b4c-42f3-b3b7-6b8fed4e2780" />

<img width="850" height="547" alt="image" src="https://github.com/user-attachments/assets/9f0905d7-295d-487d-8791-7c92a61293a2" />

<img width="850" height="547" alt="image" src="https://github.com/user-attachments/assets/376dd786-dc9a-4ae4-8e7f-d0fdb225a692" />

<img width="587" height="445" alt="image" src="https://github.com/user-attachments/assets/e8cdbd56-7668-46b6-b78f-0427c410e424" />


# Task 4: Predictive Analysis
Build a Linear Regression model to predict electricity cost (Cost_RM) based on consumption and number
of occupants.
1. Split the dataset into training and testing sets (80:20 ratio).
2. Train the linear regression model.
3. Evaluate its performance using R-squared and Mean Absolute Error (MAE).

Sample code:

<p align=center>
<img width="767" height="412" alt="image" src="https://github.com/user-attachments/assets/f3450ff9-99bb-4d89-af0a-934f321083b7" />

4. Plot Actual vs Predicted Electricity Cost and include a red dashed line (y = x) to represent perfect
prediction.

• Points above the line → over-predicted values.

• Points below the line → under-predicted values.

Sample code:

<p align=center>
<img width="759" height="188" alt="image" src="https://github.com/user-attachments/assets/5997a05b-cbdb-4ddf-9b0c-2daa69a228ce" />


**Code:**

    import numpy as np
    import pandas as pd
    import os
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_absolute_error

    # -----------------------------------------------------
    # Task 4: Predictive Analysis
    # Linear Regression Model for Electricity Cost Prediction
    # -----------------------------------------------------

    # -----------------------------------------------------
    # 1. Load the cleaned dataset
    # -----------------------------------------------------
    df = pd.read_csv('Cleaned_dataset_student.csv')

    # -----------------------------------------------------
    # 2. Define features (X) and target variable (y)
    # -----------------------------------------------------
    # Features (independent variables): Consumption_kWh and Occupants
    X = df[['Consumption_kWh', 'Occupants']]
    # Target (dependent variable): Electricity Cost
    y = df['Cost_RM']

    # -----------------------------------------------------
    # 3. Split the dataset into training and testing sets
    # -----------------------------------------------------
    # 80% of data will be used for training, 20% for testing
    # random_state ensures reproducibility of the split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------------------------------
    # 4. Train the Linear Regression model
    # -----------------------------------------------------
    # Initialize the Linear Regression model
    model = LinearRegression()
    # Train the model using the training data
    model.fit(X_train, y_train)

    # -----------------------------------------------------
    # 5. Make predictions on the test set
    # -----------------------------------------------------
    y_pred = model.predict(X_test)

    # -----------------------------------------------------
    # 6. Evaluate the model's performance
    # -----------------------------------------------------
    r2 = r2_score(y_test, y_pred) # Calculate R-squared score (coefficient of determination)
    mae = mean_absolute_error(y_test, y_pred) # Calculate Mean Absolute Error

    # Print the model performance metrics
    print('\nModel Performance:')
    print(f'R-squared: {r2:.3f}') # R-squared indicates how well the model explains the variance of the target variable
    print(f'Mean Absolute Error: {mae:.2f}') # MAE is the average absolute difference between predicted and actual values

    # -----------------------------------------------------
    # 7. Plot Actual vs Predicted values
    # -----------------------------------------------------
    plt.figure(figsize=(8, 6))
    # Scatter plot of actual vs predicted points
    plt.scatter(y_test, y_pred, color='blue', label='Predicted Points', alpha=0.6)
    # Plot a red dashed line representing perfect prediction (where y_test equals y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
          color='red', linestyle='--', linewidth=2, label='Perfect Prediction (y=x)')

    plt.xlabel('Actual Cost (RM)') # X-axis label
    plt.ylabel('Predicted Cost (RM)') # Y-axis label
    plt.title('Actual vs Predicted Electricity Cost') # Plot title
    plt.legend() # Display legend
    plt.grid(True) # Add a grid
    plt.show() # Display the plot

**Output:**

Model Performance:

R-squared: 1.000

Mean Absolute Error: 0.00

<img width="704" height="547" alt="image" src="https://github.com/user-attachments/assets/24d865ba-fe63-4b10-95ba-52f6ddb5dc97" />

# DISCUSSION (Electricity Consumption and Cost Analysis)

In this report, the analysis of electricity usage and expenditure was conducted systematically based on the information cleaning, descriptive statistics, visual presentation, and predictive modeling.

**Task 1: Data Cleaning Process**

The raw data was carefully cleaned, including missing data which was handled by interpolating the data and substituting extreme data in Consumption_kWh by the median data. Cost_RM was then re-calculated in order to maintain consistency, which created a strong platform to analyze it.

**Task 2: Data Analysis (Description)**

The descriptive statistics presented different consumption trends as Urban households had the highest consumption level of electricity, followed by Suburban and then Rural. Remarkably, there was a negative relationship that was found to be weak to moderate (-0.502) between the number of occupants and electricity consumption, which may indicate energy awareness or common resource efficiency in bigger families.

**Task 3: Data Visualization**

The hierarchy of regional consumption was verified by visualizations whereby Urban areas always topped the list. The seasonal trends in monthly trends showed significant consistency across regions. The scatter graph also helped in depicting the weak or negative correlation between occupants and consumption and confirmed that the household size does not play a crucial role in electricity use.

**Task 4: Predictive Analysis**

Linear Regression model that was created to predict Cost_RM produced superior results. The model was quite accurate as the R-squared was 1.000 and the Mean Absolute Error was 0.00 RM with the help of consumption and occupant data to predict the cost of electricity.

General finding: The discussion reveals that there are notable regional differences and seasonal factors on the consumption of electricity. The unexpected negative correlation with the occupants is an indication of complicated behavioral or efficiency variables. The forecasting model was nearly accurate which means that once consumption is known the costs associated with the same could be forecasted with high level of confidence.

**Colab Code Link:**

https://colab.research.google.com/drive/1s4UXa1KTfmOe-svlogPvhlXChMTaFGAE?usp=sharing
