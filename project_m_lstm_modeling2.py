import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# Define the window size for the moving window
window_size = 30

# Function to create sequences with a 30-day window


def create_sequences(data, window_size):
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:(i + window_size), :-1])  # All features except target
        y.append(data[i + window_size, -1])  # Target feature ('Close/Last')
    return np.array(X), np.array(y)


# Load your dataset
data = pd.read_csv(
    'C:\\Users\\lmueller\\Desktop\\Project M\\Project_M_Statistical_Data.csv')

# Segmenting the dataset by company
unique_companies = data['Company'].unique()
updated_dataframes = []

for company in unique_companies:
    print(f"Processing data for {company}...")

    # Isolate data for the current company
    company_data = data[data['Company'] == company]
    company_features = company_data.drop(['Date', 'Company'], axis=1)
    company_features = company_features.fillna(method='ffill')

    # Normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_features = scaler.fit_transform(company_features)

    # Create sequences with a 30-day window for LSTM
    X, y = create_sequences(scaled_features, window_size)

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
    predicted_prices = scaler.inverse_transform(
        np.hstack((predicted, scaled_features[window_size:, 1:])))[:, 0]

    # Add predictions to the company dataset
    # Aligning data with predictions
    company_data = company_data.iloc[window_size:]
    company_data['Prediction'] = predicted_prices

    # Append the updated company data to the list
    updated_dataframes.append(company_data)

    # Save the model for the current company
    model.save(f'{company}_model.h5')

# Combine all updated dataframes into a single dataframe
final_data = pd.concat(updated_dataframes)

# Save the final dataset with predictions for all companies
final_data.to_csv(
    'C:\\Users\\lmueller\\Desktop\\Project M\\Project_M_Statistical_Data_Predictions2.csv', index=False)
