from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load your dataset
data = pd.read_csv("resampled_data1.csv")  # Example

# Separate features (X) and target (y)
X = data.drop(columns='class')  # Replace 'class' with the actual target column name
y = data['class']  # Assuming 'class' is the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for SVM and KNN)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 1. **Linear Regression**
# Create and train the model
regressor = LinearRegression()
regressor.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred_reg = regressor.predict(X_test_scaled)
print("Linear Regression:")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_reg)}")
print(f"R2 Score: {r2_score(y_test, y_pred_reg)}\n")

# 2. **Support Vector Machine (SVR)**
# Create and train the SVR model
svr = SVR(kernel='rbf')
svr.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred_svr = svr.predict(X_test_scaled)
print("Support Vector Regression (SVR):")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_svr)}")
print(f"R2 Score: {r2_score(y_test, y_pred_svr)}\n")

# 3. **K-Nearest Neighbors (KNN) Regression**
# Create and train the KNN model
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred_knn = knn.predict(X_test_scaled)
print("K-Nearest Neighbors Regression (KNN):")
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_knn)}")
print(f"R2 Score: {r2_score(y_test, y_pred_knn)}\n")
