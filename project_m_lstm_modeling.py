import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from arch import arch_model

# Define the window size for the moving window
window_size = 30

# Function to create sequences with a 30-day window
def create_correct_sequences(data, window_size):
    X = []
    y = []
    for i in range(window_size, len(data)):
        X.append(data[i - window_size:i, :-1])
        y.append(data[i, -1])
    return np.array(X), np.array(y)

# Load your dataset
data = pd.read_csv('C:\\Users\\lennon.mueller\\Onedrive - Western Governors University\\Desktop\\Project M\\Project_M_Statistical_Data.csv')

# Check for constant columns
print("Checking for constant columns...")
for column in data.columns:
    if data[column].nunique() == 1:
        print(f"Column {column} is constant and should be removed.")

# Ensure 'Close/Last' is not in features
feature_columns = data.columns.drop(['Date', 'Company', 'Close/Last'])
assert 'Close/Last' not in feature_columns, "'Close/Last' should not be a feature."
print("Features used for training:", feature_columns)

# Segmenting the dataset by company
unique_companies = data['Company'].unique()
updated_dataframes = []

for company in unique_companies:
    print(f"Processing data for {company}...")

    company_data = data[data['Company'] == company]
    dates = company_data['Date'].values

    company_features = company_data[feature_columns]
    company_features = company_features.fillna(method='ffill')
    close_prices = company_data['Close/Last'].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(company_features)
    close_scaler = MinMaxScaler(feature_range=(0, 1)).fit(close_prices)

    scaled_features = np.hstack((scaled_features, close_scaler.transform(close_prices)))

    X, y = create_correct_sequences(scaled_features, window_size)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))

    model.compile(optimizer='adam', loss='mean_squared_error')

    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    predicted = model.predict(X)
    predicted_prices = close_scaler.inverse_transform(predicted.reshape(-1, 1))

    # Prepare the data for adjusted predictions
    company_data = company_data.iloc[window_size:]
    company_data['Original Prediction'] = predicted_prices.flatten()
    company_data['Original Prediction Change'] = company_data['Original Prediction'].diff()

    # Calculate adjusted predictions
    adjusted_predictions = []
    for index, row in company_data.iterrows():
        # Calculate dynamic mean and standard deviation for Historical_Volatility up to the current day
        historical_volatility_mean = company_data.loc[:index, 'Historical_Volatility'].mean()
        historical_volatility_std = company_data.loc[:index, 'Historical_Volatility'].std()

        # Check if Historical_Volatility is at least 1 standard deviation above the mean
        if row['Historical_Volatility'] > historical_volatility_mean + historical_volatility_std:
            adjusted_pred = row['Original Prediction'] + ((row['Historical_Volatility'] * 100) * (row['Original Prediction Change'] /10))
        else:
            adjusted_pred = row['Original Prediction']  # Use original prediction if below the threshold

        adjusted_predictions.append(adjusted_pred)

    company_data['Adjusted Prediction'] = adjusted_predictions

    updated_dataframes.append(company_data)

    model.save(f'{company}_model.h5')

final_data = pd.concat(updated_dataframes)
final_data.to_csv('C:\\Users\\lennon.mueller\\Onedrive - Western Governors University\\Desktop\\Project M\\Project_M_Statistical_Data_Predictions2.csv', index=False)
