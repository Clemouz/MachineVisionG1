import numpy as np
import laspy

# Create a grid of points representing a flat surface
x = np.linspace(0, 500, 500)
y = np.linspace(0, 500, 500)
xv, yv = np.meshgrid(x, y)
zv = np.exp(-((xv-250)**2 + (yv-250)**2)/500)
zv = zv + np.exp(-((xv-150)**2 + (yv-350)**2)/500)

# Flatten the grid to get XYZ coordinates
x_vals = xv.ravel()
y_vals = yv.ravel()
z_vals = zv.ravel()

# Set the fraction of points to keep (e.g., 70% density)
keep_fraction = 0.03

# Number of total points
num_points = x_vals.shape[0]

# Generate random mask of indices to keep
keep_indices = np.random.choice(num_points, size=int(keep_fraction * num_points), replace=False)

# Filter points by selected indices
x_vals = x_vals[keep_indices]
y_vals = y_vals[keep_indices]
z_vals = z_vals[keep_indices]

# Create LAS file header
header = laspy.LasHeader(point_format=3, version="1.2")
las = laspy.LasData(header)

# Assign XYZ coordinates
las.x = x_vals
las.y = y_vals
las.z = z_vals

# Assign classification: 2 = Ground
las.classification = np.full_like(x_vals, 2, dtype=np.uint8)

# Save to .las file
las.write("ground_surface.las")