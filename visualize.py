import laspy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the LAS or LAZ file
las = laspy.read("Points.laz")

# Extract point cloud data
x = las.x
y = las.y
z = las.z

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c=z, cmap='viridis', marker='.')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show plot
plt.show()