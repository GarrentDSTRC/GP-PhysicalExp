import numpy as np
import pandas as pd

# Given parameters
UPB=[1.0, 0.6, 40, 0, 1,1]
LOWB=[0.4, 0.1, 5, -180, -1,-1]
linspace = [10, 0,10, 0,10,0]

# Adjusting the code to insert a column of zeros for dimensions with linspace = 0
matrices = []
for i in range(len(linspace)):
    if linspace[i] > 0:
        matrices.append(np.linspace(LOWB[i], UPB[i], linspace[i]))
    else:
        matrices.append(np.zeros(1))  # Adding a column of zeros for dimensions with linspace = 0

# Create a Cartesian product of the matrices
cartesian_product = np.array(np.meshgrid(*matrices)).T.reshape(-1, len(matrices))

# Convert to DataFrame
df = pd.DataFrame(cartesian_product)
csv_file_path = 'sampled_matrix.csv'
df.to_csv(csv_file_path, index=False)

csv_file_path

