import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
import tensorflow as tf

# Load data
data = pd.read_csv("resampled_data1.csv")

# Define features and labels
X = data.drop(columns=['class'])
y = data['class']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape data for RNN input (samples, timesteps, features)
X_train_rnn = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_rnn = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Build RNN model
rnn_model = tf.keras.Sequential([
    tf.keras.layers.SimpleRNN(64, activation='relu', input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2])),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the RNN model
rnn_model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the RNN model
rnn_model.fit(X_train_rnn, y_train, epochs=10, batch_size=32, validation_data=(X_test_rnn, y_test))

# Evaluate the RNN model
rnn_loss, rnn_accuracy = rnn_model.evaluate(X_test_rnn, y_test)
print(f'RNN Test Accuracy: {rnn_accuracy}')

# Correlation Matrix
plt.figure(figsize=(10, 8))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Confusion Matrix for RNN
y_pred_rnn = (rnn_model.predict(X_test_rnn) > 0.5).astype("int32")
cm_rnn = confusion_matrix(y_test, y_pred_rnn)
disp_rnn = ConfusionMatrixDisplay(confusion_matrix=cm_rnn, display_labels=['Class 0', 'Class 1'])
disp_rnn.plot(cmap='Blues')
plt.title('Confusion Matrix - RNN')
plt.show()

# Permutation Feature Importance (PFI)
def compute_pfi(model, X, y, feature_names, metric, baseline_score=None, n_repeats=10):
    """
    Custom function to calculate Permutation Feature Importance for TensorFlow models.
    """
    if baseline_score is None:
        baseline_score = metric(y, (model.predict(X) > 0.5).astype(int).flatten())
    
    importance = []
    for i in range(X.shape[2]):  # Iterate through each feature
        X_permuted = X.copy()
        for _ in range(n_repeats):
            np.random.shuffle(X_permuted[:, :, i])
        permuted_score = metric(y, (model.predict(X_permuted) > 0.5).astype(int).flatten())
        importance.append(baseline_score - permuted_score)
    
    return importance

# Compute PFI for the RNN model
from sklearn.metrics import accuracy_score
pfi_importances = compute_pfi(rnn_model, X_test_rnn, y_test, X.columns, accuracy_score)

# Plot Permutation Feature Importance
plt.figure(figsize=(10, 6))
plt.barh(X.columns, pfi_importances, color='skyblue')
plt.xlabel("Permutation Feature Importance")
plt.ylabel("Features")
plt.title("Feature Importance - RNN (PFI)")
plt.show()
