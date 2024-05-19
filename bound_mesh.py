import numpy as np
import pandas as pd

# Given parameters
UPB=[0.25, 0.6, 40, 180, 0.95,0.95]
LOWB=[0.1, 0.1, 5, -180, -0.95,-0.95]
#linspace = [5, 0,5, 0,5,0]
linspace = [5, 5,0, 0,0,5]

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
csv_file_path = 'x.csv'
df.to_csv(csv_file_path, index=False)

csv_file_path

