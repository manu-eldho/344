import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from transformers import BartTokenizer, TFBartForSequenceClassification

# Load data
data = pd.read_csv('resampled_data1.csv')

# Define features and labels
X = data.drop(columns=['class'])
y = data['class']

# Convert `y` to integers if necessary (e.g., binary classification)
y = y.astype(int)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features for RNN and CNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for RNN input (samples, timesteps, features)
X_train_rnn = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_rnn = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# RNN Model
rnn_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(64, activation='relu', input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

rnn_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
rnn_model.fit(X_train_rnn, y_train, epochs=10, batch_size=32, validation_data=(X_test_rnn, y_test))
rnn_loss, rnn_accuracy = rnn_model.evaluate(X_test_rnn, y_test)
print(f'RNN Test Accuracy: {rnn_accuracy}')

# Reshape data for CNN input (samples, features, 1)
X_train_cnn = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
X_test_cnn = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)

# CNN Model
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Conv1D(64, 3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

cnn_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
cnn_model.fit(X_train_cnn, y_train, epochs=10, batch_size=32, validation_data=(X_test_cnn, y_test))
cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test_cnn, y_test)
print(f'CNN Test Accuracy: {cnn_accuracy}')

# Convert tabular data to text for BART
def tabular_to_text(row):
    return " ".join(f"{col}: {val}" for col, val in row.items())

X_train_text = X_train.apply(tabular_to_text, axis=1)
X_test_text = X_test.apply(tabular_to_text, axis=1)

# Tokenize the text
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
train_encodings = tokenizer(list(X_train_text), truncation=True, padding=True, max_length=128, return_tensors="tf")
test_encodings = tokenizer(list(X_test_text), truncation=True, padding=True, max_length=128, return_tensors="tf")

# BART Model
bart_model = TFBartForSequenceClassification.from_pretrained('facebook/bart-base', num_labels=2)

# Compile the BART model
bart_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the BART model
bart_model.fit(
    train_encodings.data,
    y_train.values,
    epochs=3,
    batch_size=16,
    validation_data=(test_encodings.data, y_test.values)
)

# Evaluate the BART model
bart_loss, bart_accuracy = bart_model.evaluate(test_encodings.data, y_test.values)
print(f'BART Test Accuracy: {bart_accuracy}')
