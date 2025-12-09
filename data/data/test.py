import matplotlib.pyplot as plt
import numpy as np
import h5py
import pandas as pd

# Path to the .mat file
file_path = '../data/QS04_023_data1_Elsys2.mat'

# Open the file
with h5py.File(file_path, 'r') as file:
    print("Keys: %s" % file.keys())
    data = {}
    
    # Extract and process the data
    for group_name in file.keys():
        group = file[group_name]
        for dataset_name in group.keys():
            # Retrieve the dataset
            dataset = np.array(group[dataset_name])
            
            # Check if the dataset is multi-dimensional
            if dataset.ndim > 1:
                for index, row in enumerate(dataset):
                    column_label = f'{group_name}/{dataset_name}/Row{index+1}'
                    # Create a pandas Series for each row
                    data[column_label] = pd.Series(row)
            else:
                # If the dataset is 1D, store it directly as a Series
                column_label = f'{group_name}/{dataset_name}'
                data[column_label] = pd.Series(dataset)

# Convert the dictionary of Series to a DataFrame
df = pd.DataFrame(data)

# Print the DataFrame structure to check
print(df.head())


# Save to CSV
csv_path = file_path.replace('.mat', '.csv')
df.to_csv(csv_path, index=False)
print(f'Data saved to {csv_path}')
