import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load the dataset
file_path = r'C:\Users\lmueller\Desktop\Project M\Updated_Dataset_with_Indicators_Test2.csv'
data = pd.read_csv(file_path)

# Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Update the indicators list to exclude price-based indicators like 50D MA and 200D MA
indicators = ['RSI', 'MACD', 'MACD_Signal', 'Volume', 'OBV',
              'Bollinger_Upper', 'Bollinger_Lower', 'ATR', 'ROC']

# Function to identify growth periods


def identify_growth_periods(company_data):
    growth_periods = []
    for i in range(len(company_data) - 22):
        start_price = company_data.iloc[i]['Close/Last']
        end_price = company_data.iloc[i + 22]['Close/Last']
        if ((end_price - start_price) / start_price) >= 0.25:
            growth_periods.append(i)
    return growth_periods

# Calculate average indicator values at the start of growth periods


def calc_avg_indicators_at_growth_starts(company_data, growth_periods):
    avg_indicators = company_data.iloc[growth_periods][indicators].mean()
    return avg_indicators


# Main processing
results = []

for company in data['Company'].unique():
    company_data = data[data['Company'] == company].reset_index(drop=True)

    # Fill NaN values in the numerical columns before scaling
    numerical_means = company_data[indicators].mean()
    company_data[indicators] = company_data[indicators].fillna(numerical_means)

    growth_periods = identify_growth_periods(company_data)
    if not growth_periods:
        continue

    avg_indicators = calc_avg_indicators_at_growth_starts(
        company_data, growth_periods)

    # Standardize indicators
    scaler = StandardScaler()
    company_data_scaled = scaler.fit_transform(company_data[indicators])
    avg_scaled = scaler.transform([avg_indicators])

    # Calculate similarity (Euclidean distance) and rank days
    distances = [euclidean(day, avg_scaled[0]) for day in company_data_scaled]
    company_data['Similarity Rank'] = distances
    company_data['Similarity Rank'] = company_data['Similarity Rank'].rank(
        method='min')

    # Determine the 8th percentile rank
    threshold_rank = np.percentile(company_data['Similarity Rank'], 5)

    # Flag days within the top 92% similarity as potential buy signals
    company_data['30 Day Buy Signal'] = (
        company_data['Similarity Rank'] <= threshold_rank).astype(int)

    results.append(company_data)

# Combine results and save to CSV
final_results = pd.concat(results).sort_values(by=['Company', 'Date'])
final_results.to_csv(
    r'C:\Users\lmueller\Desktop\Project M\Updated_Dataset_with_Signals_Ranked.csv', index=False)
