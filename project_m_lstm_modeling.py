import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Define the window size for the moving window
window_size = 30

# Function to create sequences with a 30-day window


def create_correct_sequences(data, window_size):
    X = []
    y = []
    for i in range(window_size, len(data)):
        # All features except target for the past 30 days
        X.append(data[i - window_size:i, :-1])
        y.append(data[i, -1])  # Target feature ('Close/Last') for the next day
    return np.array(X), np.array(y)


# Load your dataset
data = pd.read_csv(
    'C:\\Users\\lmueller\\Desktop\\Project M\\Project_M_Statistical_Data.csv')

# Debugging: Check for constant columns
print("Checking for constant columns...")
for column in data.columns:
    if data[column].nunique() == 1:
        print(f"Column {column} is constant and should be removed.")

# Debugging: Ensure 'Close/Last' is not in features
feature_columns = data.columns.drop(['Date', 'Company', 'Close/Last'])
assert 'Close/Last' not in feature_columns, "'Close/Last' should not be a feature."
print("Features used for training:", feature_columns)

# Segmenting the dataset by company
unique_companies = data['Company'].unique()
updated_dataframes = []

for company in unique_companies:
    print(f"Processing data for {company}...")

    # Isolate data for the current company
    company_data = data[data['Company'] == company]
    dates = company_data['Date'].values  # Store the dates for debugging

    # Separate features and target
    company_features = company_data[feature_columns]
    company_features = company_features.fillna(method='ffill')
    close_prices = company_data['Close/Last'].values.reshape(-1, 1)

    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(company_features)
    close_scaler = MinMaxScaler(feature_range=(0, 1)).fit(close_prices)

    # Combine the scaled features with the 'Close/Last' prices
    # Ensure that the target ('Close/Last') is at the end of the feature set
    scaled_features = np.hstack(
        (scaled_features, close_scaler.transform(close_prices)))

    # Create sequences with a 30-day window for LSTM
    X, y = create_correct_sequences(scaled_features, window_size)

    # Splitting the data (ensure no future data in the test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False)

    # Building the LSTM model for the current company
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True,
              input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output layer

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Training the model
    model.fit(X_train, y_train, epochs=50, batch_size=32,
              validation_data=(X_test, y_test))

    # Making predictions on the entire dataset (scaled features)
    predicted = model.predict(X)

    # Inverse scaling of predictions
    predicted_prices = close_scaler.inverse_transform(predicted.reshape(-1, 1))

    # Add predictions to the company dataset and aligning data with predictions
    company_data = company_data.iloc[window_size:]
    company_data['Prediction'] = predicted_prices.flatten()

    # Debugging: Print date ranges and actual vs predicted prices for the first few predictions
    for i in range(10):  # Adjust the range as needed for more predictions
        print(f"Predicting for date: {dates[window_size + i]}")
        print(f"Using data from: {dates[i]} to {dates[i + window_size - 1]}")
        print(
            f"Actual price: {company_data.iloc[i]['Close/Last']}, Predicted price: {predicted_prices[i][0]}\n")

    # Append the updated company data to the list
    updated_dataframes.append(company_data)

    # Save the model for the current company
    model.save(f'{company}_model.h5')

# Combine all updated dataframes into a single dataframe
final_data = pd.concat(updated_dataframes)

# Save the final dataset with predictions for all companies
final_data.to_csv(
    'C:\\Users\\lmueller\\Desktop\\Project M\\Project_M_Statistical_Data_Predictions2.csv', index=False)
