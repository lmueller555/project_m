import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

# File path for the Excel file
file_path = 'C:\\Users\\lmueller\\Desktop\\Project M\\Compiled_Companies_Data_with_MA.xlsx'

# Reading the data from the Excel file
dataset = pd.read_excel(file_path)

# Convert 'Date' to datetime and sort by date
dataset['Date'] = pd.to_datetime(dataset['Date'])
dataset.sort_values('Date', inplace=True)

# Selecting features
features = ['Close/Last', 'Open', 'Daily High', 'Daily Low', '200d MA']

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))

# Sequence length
sequence_length = 60

# Dictionary to store scalers for each company
scalers = {}

# Dictionary to store predictions
all_predictions = {}

# Loop over each company
for company in dataset['Company'].unique():
    print(f"Processing {company}")

    # Filter data for the current company
    company_data = dataset[dataset['Company'] == company][features]

    # Normalize the company data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(company_data)
    scalers[company] = scaler  # Store the scaler for inverse scaling later

    # Generate training and testing data for the company
    X, y = [], []
    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i - sequence_length:i])
        y.append(scaled_data[i, 0])  # Assuming 'Close/Last' is at index 0

    X = np.array(X)
    y = np.array(y)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Check if a model already exists for this company
    model_filename = f'best_lstm_model_{company}.h5'
    if not os.path.exists(model_filename):
        # Build LSTM model for the company
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True,
                  input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))  # Predicting 'Close/Last'

        model.compile(optimizer='adam', loss='mean_squared_error')

        # Callbacks for early stopping and model saving
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)
        model_checkpoint = ModelCheckpoint(
            model_filename, monitor='val_loss', save_best_only=True)

        # Train the model for the company
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(
            X_test, y_test), callbacks=[early_stopping, model_checkpoint])
        print(f"Model for {company} trained and saved as '{model_filename}'")
    else:
        # Load the existing model
        model = tf.keras.models.load_model(model_filename)
        print(f"Model for {company} loaded from '{model_filename}'")

    # Prepare data for prediction (reshape into sequence format)
    prediction_data = []
    for i in range(len(scaled_data) - sequence_length):
        seq = scaled_data[i:i + sequence_length]
        if seq.shape[0] == sequence_length:
            prediction_data.append(seq)

    prediction_data = np.array(prediction_data)

    # Generate predictions for the entire dataset for this company
    if prediction_data.shape[0] > 0:
        company_predictions = model.predict(prediction_data)

        # Inverse scale the predictions to get actual values
        company_predictions = scaler.inverse_transform(np.hstack((company_predictions, np.zeros(
            (company_predictions.shape[0], scaled_data.shape[1] - 1)))))[:, 0]
        all_predictions[company] = company_predictions
    else:
        print(f"No prediction data available for {company}")

# Creating a DataFrame from the predictions dictionary
predictions_df = pd.DataFrame(all_predictions)

# Add the Date column from the original dataset
dates = dataset['Date'][sequence_length:].reset_index(drop=True)
predictions_df.insert(0, 'Date', dates)

# Generating a new filename for the predictions file
original_filename = os.path.basename(file_path)
new_filename = original_filename.replace('.xlsx', '_Predictions.csv')

# Save to CSV
predictions_file_path = os.path.join(os.path.dirname(file_path), new_filename)
predictions_df.to_csv(predictions_file_path, index=False)

print(f"Predictions saved to {predictions_file_path}")
