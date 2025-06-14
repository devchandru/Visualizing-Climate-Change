"""
Visualizing Climate Change Project

"""


# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


# Set global parameters and styles

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
np.random.seed(42)


# Generate Simulated Climate Data

# Generate years
years = np.arange(1960, 2021)

# Simulate CO₂ Concentration data (in ppm)
co2_base = 315  # starting ppm for the year 1960
# Cumulative sum of random normal increments to simulate gradual increase
co2_increments = np.random.normal(1.5, 0.2, len(years))
co2_concentration = co2_base + np.cumsum(co2_increments)

# Simulate Temperature Anomaly data (in °C)
# Start slightly below zero and trend upward with noise
temp_anomaly = -0.2 + 0.02 * (years - 1960) + np.random.normal(0, 0.1, len(years))

# Create a DataFrame for climate data
climate_df = pd.DataFrame({
    'Year': years,
    'CO2_Concentration': co2_concentration,
    'Temperature_Anomaly': temp_anomaly
})


# Save the simulated data to CSV (for record keeping)

csv_path = "climate_change_data.csv"
climate_df.to_csv(csv_path, index=False)
print(f"Data saved to: {csv_path}")


# Exploratory Data Analysis and Summary Statistics

def print_statistics(data):
    """
    Print summary statistics of the climate data.
    """
    print("----- Climate Data Overview -----")
    print(data.describe())
    print("---------------------------------\n")

print_statistics(climate_df)


# Basic Trend Plots

def plot_co2_trend(data):
    """Plot CO₂ concentration trend over time."""
    plt.figure()
    plt.plot(data['Year'], data['CO2_Concentration'], label='CO₂ Concentration (ppm)',
             color='green', marker='o', linestyle='-')
    plt.title("Atmospheric CO₂ Concentration Over Time")
    plt.xlabel("Year")
    plt.ylabel("CO₂ Concentration (ppm)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("climate_co2_trend.png")
    plt.close()
    print("CO₂ trend plot saved as 'climate_co2_trend.png'")

def plot_temperature_trend(data):
    """Plot global temperature anomaly trend over time."""
    plt.figure()
    plt.plot(data['Year'], data['Temperature_Anomaly'], label='Temperature Anomaly (°C)',
             color='red', marker='o', linestyle='-')
    plt.title("Global Temperature Anomaly Over Time")
    plt.xlabel("Year")
    plt.ylabel("Temperature Anomaly (°C)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("climate_temp_trend.png")
    plt.close()
    print("Temperature trend plot saved as 'climate_temp_trend.png'")

plot_co2_trend(climate_df)
plot_temperature_trend(climate_df)


# Dual Axis Plot

def plot_dual_axis(data):
    """
    Create a dual-axis plot for CO₂ concentration and Temperature Anomaly.
    """
    fig, ax1 = plt.subplots()

    color1 = 'tab:green'
    ax1.set_xlabel('Year')
    ax1.set_ylabel('CO₂ Concentration (ppm)', color=color1)
    ax1.plot(data['Year'], data['CO2_Concentration'], color=color1, marker='o')
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel('Temperature Anomaly (°C)', color=color2)
    ax2.plot(data['Year'], data['Temperature_Anomaly'], color=color2, marker='o')
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title("Dual Axis: CO₂ vs Temperature Anomaly")
    fig.tight_layout()
    plt.savefig("dual_axis_plot.png")
    plt.close()
    print("Dual axis plot saved as 'dual_axis_plot.png'")

plot_dual_axis(climate_df)


# Correlation Analysis and Heatmap

def plot_correlation_heatmap(data):
    """
    Compute and display the correlation heatmap between CO₂ concentration
    and Temperature Anomaly.
    """
    corr = data[['CO2_Concentration', 'Temperature_Anomaly']].corr()
    plt.figure()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of Climate Data")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()
    print("Correlation heatmap saved as 'correlation_heatmap.png'")
    return corr

correlation_matrix = plot_correlation_heatmap(climate_df)
print("Correlation Matrix:\n", correlation_matrix)


# Rate of Change Calculation

def calculate_rate_of_change(data):
    """
    Calculate yearly rate of change for CO₂ concentration and Temperature Anomaly.
    """
    data['CO2_Change'] = data['CO2_Concentration'].diff()
    data['Temp_Change'] = data['Temperature_Anomaly'].diff()
    return data

climate_df = calculate_rate_of_change(climate_df)

def plot_rate_of_change(data):
    """
    Plot the rate of change (yearly difference) for CO₂ and Temperature Anomaly.
    """
    plt.figure()
    plt.plot(data['Year'], data['CO2_Change'], label="Δ CO₂ Concentration", color='blue', marker='o')
    plt.plot(data['Year'], data['Temp_Change'], label="Δ Temperature Anomaly", color='orange', marker='o')
    plt.xlabel("Year")
    plt.ylabel("Yearly Change")
    plt.title("Yearly Rate of Change in Climate Data")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("rate_of_change_plot.png")
    plt.close()
    print("Rate of change plot saved as 'rate_of_change_plot.png'")

plot_rate_of_change(climate_df)


# Moving Average Smoothing

def moving_average(data, window=5):
    """
    Calculate moving average for smoothing time-series data.
    """
    ma_data = data.rolling(window=window).mean()
    return ma_data

ma_climate_df = moving_average(climate_df[['CO2_Concentration', 'Temperature_Anomaly']])
# Plot moving averages
plt.figure()
plt.plot(climate_df['Year'], climate_df['CO2_Concentration'], label="CO₂ Original", color='green', alpha=0.4)
plt.plot(climate_df['Year'], ma_climate_df['CO2_Concentration'], label="CO₂ MA", color='green', linestyle="--")
plt.plot(climate_df['Year'], climate_df['Temperature_Anomaly'], label="Temp Original", color='red', alpha=0.4)
plt.plot(climate_df['Year'], ma_climate_df['Temperature_Anomaly'], label="Temp MA", color='red', linestyle="--")
plt.xlabel("Year")
plt.ylabel("Value")
plt.title("Moving Average Smoothing (Window=5)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("moving_average_plot.png")
plt.close()
print("Moving average plot saved as 'moving_average_plot.png'")


# Polynomial Regression Analysis

def polynomial_regression(data, degree=3):
    """
    Fit a polynomial regression model to the data and return predictions.
    """
    # Prepare data
    X = data['Year'].values.reshape(-1, 1)
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    
    # Fit model for CO2
    model_co2 = LinearRegression()
    model_co2.fit(X_poly, data['CO2_Concentration'])
    co2_pred = model_co2.predict(X_poly)

    # Fit model for Temperature Anomaly
    model_temp = LinearRegression()
    model_temp.fit(X_poly, data['Temperature_Anomaly'])
    temp_pred = model_temp.predict(X_poly)
    
    return X, co2_pred, temp_pred, poly

X_poly, co2_pred, temp_pred, poly_model = polynomial_regression(climate_df, degree=3)

# Plot polynomial regression
plt.figure()
plt.scatter(climate_df['Year'], climate_df['CO2_Concentration'], color='green', label="CO₂ Data", alpha=0.6)
plt.plot(climate_df['Year'], co2_pred, color='darkgreen', label="CO₂ Poly Fit")
plt.scatter(climate_df['Year'], climate_df['Temperature_Anomaly'], color='red', label="Temp Data", alpha=0.6)
plt.plot(climate_df['Year'], temp_pred, color='darkred', label="Temp Poly Fit")
plt.xlabel("Year")
plt.ylabel("Value")
plt.title("Polynomial Regression (Degree=3)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("polynomial_regression_plot.png")
plt.close()
print("Polynomial regression plot saved as 'polynomial_regression_plot.png'")

# KMeans Clustering on Normalized Data

def clustering_analysis(data, n_clusters=3):
    """
    Perform KMeans clustering on the climate data.
    """
    # Normalize the features
    scaler = StandardScaler()
    features = data[['CO2_Concentration', 'Temperature_Anomaly']].dropna()
    norm_features = scaler.fit_transform(features)
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(norm_features)
    
    # Add cluster labels to DataFrame
    data_clustered = features.copy()
    data_clustered['Cluster'] = clusters
    return data_clustered, kmeans

climate_clusters, kmeans_model = clustering_analysis(climate_df)

# Plot clustering results
plt.figure()
sns.scatterplot(x=climate_clusters.index, y=climate_clusters['CO2_Concentration'],
                hue=climate_clusters['Cluster'], palette="viridis", s=100)
plt.title("KMeans Clustering on CO₂ Data")
plt.xlabel("Index")
plt.ylabel("CO₂ Concentration (ppm)")
plt.tight_layout()
plt.savefig("clustering_co2_plot.png")
plt.close()
print("Clustering plot saved as 'clustering_co2_plot.png'")

# Additional Analysis: Yearly Trend and Regression Slope

def regression_slope(data):
    """
    Calculate and print the regression slope (rate) for CO₂ and Temperature.
    """
    X = data['Year'].values.reshape(-1,1)
    reg_co2 = LinearRegression().fit(X, data['CO2_Concentration'])
    reg_temp = LinearRegression().fit(X, data['Temperature_Anomaly'])
    slope_co2 = reg_co2.coef_[0]
    slope_temp = reg_temp.coef_[0]
    print(f"Regression Slope for CO₂: {slope_co2:.3f} ppm per year")
    print(f"Regression Slope for Temperature Anomaly: {slope_temp:.3f} °C per year")
    
regression_slope(climate_df)

# Saving Final Processed Data with Additional Columns

final_csv_path = "climate_change_processed_data.csv"
climate_df.to_csv(final_csv_path, index=False)
print(f"Final processed data saved as: {final_csv_path}")

# End of Project

print("\nVisualizing Climate Change Project Complete!")
print("All plots and data files have been saved in the current directory.")
print("\nThank you for using the Climate Change Visualization project.")

