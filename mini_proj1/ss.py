from imblearn.over_sampling import SMOTE
import pandas as pd

# Load your dataset
data = pd.read_csv("C:\\Users\\manu eldho\\Downloads\\Datanew_feat_select (1).csv")  # Example

# Separate features (X) and target (y)
X = data.drop(columns='class')  # Replace 'target' with your actual target column name
y = data['class']

# Get the current class distribution
class_counts = y.value_counts()

# Calculate the desired number of samples for each class
desired_samples = 10000

# Set the sampling strategy as a dictionary with the number of samples per class
sampling_strategy = {class_: desired_samples for class_ in class_counts.index}

# Apply SMOTE with the calculated sampling strategy
smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)

# Apply SMOTE to generate synthetic samples
X_resampled, y_resampled = smote.fit_resample(X, y)

# Combine resampled features and target back into a DataFrame
resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
resampled_data['class'] = y_resampled  # Ensure the target column name matches your data

# Write the resampled data to a new CSV file
resampled_data.to_csv("resampled_data1.csv", index=False)

# Check the class distribution in the resampled data
print(resampled_data['class'].value_counts())
