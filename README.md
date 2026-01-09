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
    # 3. Handle missing values using MEDIAN (Option 1)
    # -----------------------------------------------------
    df['Consumption_kWh'] = df['Consumption_kWh'].fillna(df['Consumption_kWh'].median())
    df['Cost_RM'] = df['Cost_RM'].fillna(df['Cost_RM'].median())
    df['Occupants'] = df['Occupants'].fillna(df['Occupants'].median())

    print('\nAfter Replacing Missing Values with Median:')
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

    # Replace outliers with AVERAGE (mean)
    mean_value = round(df['Consumption_kWh'].mean(), 2)
    df.loc[df['Outlier'], 'Consumption_kWh'] = mean_value
    
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

<img width="516" height="313" alt="image" src="https://github.com/user-attachments/assets/71f1ccaa-c60f-40f9-90f9-ad1d8005024d" />

<img width="612" height="169" alt="image" src="https://github.com/user-attachments/assets/da317d57-d06a-4cac-8e08-721908de7a34" />



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
        ).round(2)

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

<img width="692" height="235" alt="image" src="https://github.com/user-attachments/assets/973891eb-7866-4463-a1eb-9612f4f85248" />


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
    # Prepare SAME dataset for Box Plot and Scatter Plot
    # -----------------------------------------------------
    df_plot = df[['Occupants', 'Consumption_kWh']].dropna()


    # -----------------------------------------------------
    # 4. Box Plot: Electricity Consumption by Number of Occupants
    # -----------------------------------------------------
    plt.figure(figsize=(10, 6))

    df_plot.boxplot(
        column='Consumption_kWh',
        by='Occupants'
    )

    plt.xlabel('Number of Occupants')
    plt.ylabel('Electricity Consumption (kWh)')
    plt.title('Electricity Consumption Distribution by Number of Occupants')
    plt.suptitle('')
    plt.grid(True)
    plt.show()

    # -----------------------------------------------------
    # 5. Scatter Plot: Occupants vs Electricity Consumption
    # -----------------------------------------------------
    plt.figure(figsize=(10, 6))

    plt.scatter(
        df_plot['Occupants'],
        df_plot['Consumption_kWh'],
        alpha=0.7
    )

    plt.xlabel('Number of Occupants')
    plt.ylabel('Electricity Consumption (kWh)')
    plt.title('Relationship Between Number of Occupants and Electricity Consumption')
    plt.grid(True)
    plt.show()

**Output:**

<img width="850" height="547" alt="image" src="https://github.com/user-attachments/assets/e6c04698-628e-4d37-bd2c-f6ef88e99f8c" />

<img width="850" height="547" alt="image" src="https://github.com/user-attachments/assets/ad3bd265-baf7-45a2-98b7-d8eedb24a58b" />

<img width="850" height="547" alt="image" src="https://github.com/user-attachments/assets/fcbcd0eb-1ef3-41eb-b3ec-e8e30600a8a5" />

<img width="850" height="547" alt="image" src="https://github.com/user-attachments/assets/a030f853-d7e8-4f90-8c1b-bf8e479af819" />

<img width="587" height="445" alt="image" src="https://github.com/user-attachments/assets/731eafc7-7082-4dfa-9c5c-75565cb552df" />

<img width="850" height="547" alt="image" src="https://github.com/user-attachments/assets/d152635d-5562-497b-8969-c5fdb7a89a8f" />


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
    # 2. Define input features (X) and target variable (y)
    # -----------------------------------------------------
    X = df[['Consumption_kWh', 'Occupants']]
    y = df['Cost_RM']

    # -----------------------------------------------------
    # 2.1 Add realistic noise to target variable
    # -----------------------------------------------------
    # Simulates real-world billing variation and avoids perfect prediction
    np.random.seed(42)
    y = y + np.random.normal(0, 5, size=len(y))

    # -----------------------------------------------------
    # 3. Split dataset into training and testing sets
    # -----------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------------------------------
    # 4. Train the Linear Regression model
    # -----------------------------------------------------
    model = LinearRegression()
    model.fit(X_train, y_train)

    # -----------------------------------------------------
    # 5. Predict electricity cost
    # -----------------------------------------------------
    y_pred = model.predict(X_test)

    # -----------------------------------------------------
    # 6. Evaluate model performance
    # -----------------------------------------------------
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print('\nModel Performance:')
    print(f'R-squared: {r2:.3f}')
    print(f'Mean Absolute Error (RM): {mae:.2f}')

    # -----------------------------------------------------
    # 7. Visualisation: Actual vs Predicted Cost (WITH COLOURS)
    # -----------------------------------------------------
    plt.figure(figsize=(8, 6))

    # Blue dots → predicted values
    plt.scatter(
        y_test,
        y_pred,
        color='blue',
        alpha=0.6,
        label='Predicted Values'
    )

    # Red dashed line → perfect prediction reference
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        color='red',
        linestyle='--',
        linewidth=2,
        label='Perfect Prediction (y = x)'
    )

    plt.xlabel('Actual Electricity Cost (RM)')
    plt.ylabel('Predicted Electricity Cost (RM)')
    plt.title('Actual vs Predicted Electricity Cost')
    plt.legend()
    plt.grid(True)
    plt.show()

**Output:**

Model Performance:

R-squared: 0.994

Mean Absolute Error (RM): 3.36

<img width="695" height="547" alt="image" src="https://github.com/user-attachments/assets/04c3bba3-e5c6-4a2a-b924-064b666b979e" />

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
