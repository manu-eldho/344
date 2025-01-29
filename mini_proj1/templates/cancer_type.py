import pandas as pd
import numpy as np

# Define the number of samples
n_samples = 100

# Generate synthetic data for original features
data = {
    'PP': np.random.choice([2, 4, 6], n_samples),
    'SIFTR': np.random.uniform(0.4, 0.9, n_samples),
    'Polyphen2R': np.random.uniform(0.4, 0.9, n_samples),
    'Polyphen2P': np.random.choice([-1, 1, 2], n_samples),
    'PROVEANR': np.random.uniform(0.2, 0.8, n_samples),
    'PROVEANP': np.random.choice([0, 1], n_samples),
    'CADDS': np.random.uniform(2.5, 4.5, n_samples),
    'CADDR': np.random.uniform(0.4, 0.9, n_samples),
    'CADDP': np.random.uniform(20, 30, n_samples),
    'fathmmS': np.random.uniform(0.4, 0.8, n_samples),
    'fathmmR': np.random.uniform(0.2, 0.4, n_samples),
    'fathmmP': np.random.choice([0, 1], n_samples),
    'phyloP': np.random.uniform(0.02, 4.5, n_samples),
    'phyloPR': np.random.uniform(0.1, 0.6, n_samples),
    'Class': np.random.choice([0, 1], n_samples),  # Retain class label
}

# Add new features
data['Tissue_Type'] = np.random.choice(['Colon', 'Breast', 'Prostate', 'Lung'], n_samples)
data['TP53_Mutation'] = np.random.choice([0, 1], n_samples)
data['KRAS_Mutation'] = np.random.choice([0, 1], n_samples)
data['BRCA1_Mutation'] = np.random.choice([0, 1], n_samples)
data['POLB_Expression'] = np.random.uniform(0.5, 1.5, n_samples)

# Assign cancer type based on tissue type and mutations
def assign_cancer_type(row):
    if row['Tissue_Type'] == 'Colon':
        return 'Colorectal'
    elif row['Tissue_Type'] == 'Breast':
        return 'Breast'
    elif row['Tissue_Type'] == 'Prostate':
        return 'Prostate'
    elif row['Tissue_Type'] == 'Lung':
        return 'Lung'
    else:
        return 'Unknown'

data['Cancer_Type'] = [assign_cancer_type(row) for row in pd.DataFrame(data).to_dict('records')]

# Create DataFrame
df = pd.DataFrame(data)

# Save the dataset to a CSV file
output_file = "synthetic_cancer_dataset.csv"
df.to_csv(output_file, index=False)

print(f"Synthetic dataset saved to {output_file}")