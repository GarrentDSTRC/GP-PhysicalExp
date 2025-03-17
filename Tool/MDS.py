from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import os
normalize=0
i = 13
def replace_negative_values_with_minus_one(data):
    #data.iloc[:, -1]=data.iloc[:, -1] / 16
    # Replace negative values in the last column with -1
    data.iloc[:, -1] = np.where(data.iloc[:, -1] < 0,data.iloc[:, -1] / 100, data.iloc[:, -1])
    return data

def save():
    output_dir = "save"
    output_filename = "mds_reduced_data.csv"
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Combine the reduced data and output
    combined_data = pd.DataFrame(reduced_data, columns=['MDS_Dim1', 'MDS_Dim2'])
    combined_data['Output'] = data.iloc[:, -1]
    # Save the combined data to a CSV file
    combined_data.to_csv(os.path.join(output_dir, output_filename), index=False)

# Load the data from the Excel file
data = pd.read_excel('train_x.xlsx')

# Replace NaN values in the output column with the mean of the non-NaN values
data.fillna(data.iloc[:, -1].mean(), inplace=True)
#replace_negative_values_with_minus_one(data)

# Apply MDS to reduce the input data to two dimensions
mds = MDS(n_components=2, random_state=42)
reduced_data = mds.fit_transform(data.iloc[:, :-1])

# Create a 3D plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the reduced data points with their output values
ax.scatter(reduced_data[:, 0], reduced_data[:, 1], data.iloc[:, -1], c='r', marker='o')

# Set labels for the axes
ax.xaxis.set_tick_params(labelsize=i)
ax.yaxis.set_tick_params(labelsize=i)
ax.zaxis.set_tick_params(labelsize=i)
ax.set_xlabel(r'$p_1$', fontsize=i)
ax.set_ylabel(r'$p_2$', fontsize=i)
ax.set_zlabel('Output', fontsize=i)

plt.title('3D Scatter Plot of MDS Reduced Data')

save()


# Create a 2D plot with the scatter plot
fig, ax = plt.subplots(figsize=(10, 8))
# Plot the reduced data points with their output values, using a color map based on the output
scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=data.iloc[:, -1], cmap='jet', alpha=0.7)
# Set labels for the axes
ax.xaxis.set_tick_params(labelsize=i)
ax.yaxis.set_tick_params(labelsize=i)
ax.set_xlabel(r'$p_1$', fontsize=i)
ax.set_ylabel(r'$p_2$', fontsize=i)
# Add color bar to the plot
plt.colorbar(scatter, ax=ax)
cbar = ax.collections[0].colorbar
# 设置颜色条的刻度字体大小
cbar.ax.tick_params(labelsize=i)
#plt.title('2D Scatter Plot of MDS Reduced Data with Jet Color Mapping')

from scipy.interpolate import interp1d
from scipy.interpolate import griddata
# Create a grid of points
x = np.linspace(reduced_data[:, 0].min(), reduced_data[:, 0].max(), 50)
y = np.linspace(reduced_data[:, 1].min(), reduced_data[:, 1].max(), 50)
x, y = np.meshgrid(x, y)

# Interpolate the z values using griddata
z = griddata((reduced_data[:, 0], reduced_data[:, 1]), data.iloc[:, -1], (x, y), method='linear')

if normalize:
    z_normalized_filtered = z[~np.isnan(z)]
    # 找到过滤后的z值中的极大和极小值
    z_min = np.min(z_normalized_filtered)
    z_max = np.max(z_normalized_filtered)
    z = (z - z_min) / (z_max - z_min)

# Create a 3D plot with the surface plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')

# Set labels for the axes
ax.xaxis.set_tick_params(labelsize=i)
ax.yaxis.set_tick_params(labelsize=i)
ax.zaxis.set_tick_params(labelsize=i)
ax.set_xlabel(r'$p_1$', fontsize=i)
ax.set_ylabel(r'$p_2$', fontsize=i)
ax.set_zlabel(r'$\alpha$',fontsize=i)

plt.title('3D Surface Plot of MDS Reduced Data')
plt.show()



# 创建等高线图
fig, ax = plt.subplots(figsize=(8, 6))

# 设置等高线的水平，从-5到1，例如每隔1设置一个等高线
#levels = np.linspace(-40, 40, 9)
levels = [-0.9,-0.3,0,0.1,0.2,0.25,0.3]
#levels = [0.05,0.1,0.15,0.2,0.25,0.3]
# contourf()用于绘制填充的等高线图
contour = ax.contourf(x, y, z, levels=levels, cmap='viridis')
#contour = ax.contourf(x, y, z,  cmap='viridis')

# 添加颜色条
cbar = plt.colorbar(contour, ax=ax)
cbar.ax.tick_params(labelsize=i)
ax.tick_params(axis='both', which='major', labelsize=i)

# 设置坐标轴标签
ax.set_xlabel(r'$p_1$', fontsize=i)
ax.set_ylabel(r'$p_2$', fontsize=i)

# 设置标题
#plt.title('Contour Plot of MDS Reduced Data with Levels Ranging from -5 to 1')

# 显示图形
plt.show()
# ...（省略上面的代码）
from scipy.spatial import cKDTree

def find_nearest_values_batch(X, Y, Z, x, y, z, values):
    """
    Find the nearest values in the 'values' array for points in 'X', 'Y', 'Z' in a batch process.

    Parameters:
    X, Y, Z (numpy arrays): The coordinates of the points for which to find the nearest values.
    x, y, z (numpy arrays): The coordinates of the points with known values.
    values (numpy array): The values associated with the points 'x', 'y', 'z'.

    Returns:
    numpy array: An array of the nearest values for the points 'X', 'Y', 'Z'.
    """

    # Combine the x, y, z coordinates into a single array for the KD tree
    points = np.column_stack((x, y, z))
    # Create a KD tree for efficient nearest neighbor search
    tree = cKDTree(points)

    # Combine the X, Y, Z coordinates into a single array for querying
    query_points = np.column_stack((X, Y, Z))

    # Find the indices of the nearest points in the KD tree
    distances, indices = tree.query(query_points)

    # Retrieve the corresponding values for the nearest points
    nearest_values = values[indices]

    return nearest_values

# Example usage:
# Assuming x, y, z, values are already defined as in your previous example
#nearest_values = find_nearest_values_batch(X, Y, Z, x, y, z, values)

# 检查最后一列的长度是否为3，如果是，则绘制3维等高线图
if data.iloc[0, :-1].shape[0] == 3:
    import plotly.graph_objects as go
    import numpy as np

    # 假设data是一个pandas DataFrame，其中前三列是x、y、z，最后一列是out
    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values
    z = data.iloc[:, 2].values
    values = data.iloc[:, 3].values

    x1 = np.linspace(0.1, 0.25, 5)
    y1 = np.linspace(0.1, 0.6, 5)
    z1 = np.linspace(-0.95, 0.95, 5)
    X, Y, Z = np.meshgrid(x1, y1, z1)
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()
    VALUES=find_nearest_values_batch(X, Y, Z, x, y, z, values)
    # 创建3D表面图
    #VALUES= X+Y+Z

    fig = go.Figure(data=go.Isosurface(x=X, y=Y, z=Z, value=VALUES,
                                       isomin=min(VALUES),
                                       isomax=max(VALUES) ,
                                       # surface_fill=0.7,
                                       # opacity=0.9,  # 改变图形的透明度
                                       colorscale='jet',  # 改变颜色

                                       surface_count=5,
                                       colorbar_nticks=7,
                                       caps=dict(x_show=False, y_show=False, z_show=False),
                    ))

    #设置坐标轴标签
    fig.update_layout(scene=dict(
        xaxis_title=r'St',
        yaxis_title=r'Heave',
        zaxis_title=r'TW',
        xaxis_tickfont_size=13,
        yaxis_tickfont_size=13,
        zaxis_tickfont_size=13,
        xaxis_titlefont_size=18,
        yaxis_titlefont_size=18,
        zaxis_titlefont_size=18,
        xaxis_nticks=5,
        yaxis_nticks=5,
    ))

    # 显示图形
    fig.show()


