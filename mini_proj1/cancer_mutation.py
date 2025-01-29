import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Load dataset
data = pd.read_csv("resampled_data1.csv")

# Convert data to NumPy array
X = data.iloc[:, :-1].values  # All columns except the first (ID) and last (class)
y = data.iloc[:, -1].values    # Last column (class)

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save the scaler to a file for later use in the Flask app
joblib.dump(scaler, 'scaler.pkl')

# Reshape for RNN: [samples, timesteps=1, features]
X = np.expand_dims(X, axis=1)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Define RNN model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# Save the trained model
model.save("rnn_model.h5")

#print("Scaler and model saved successfully.")
