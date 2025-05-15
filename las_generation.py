import numpy as np
import laspy

# Create a grid of points representing a flat surface
x = np.linspace(0, 100, 100)
y = np.linspace(0, 100, 100)
xv, yv = np.meshgrid(x, y)
zv = np.exp(-((xv-50)**2 + (yv-50)**2)/100)

# Flatten the grid to get XYZ coordinates
x_vals = xv.ravel()
y_vals = yv.ravel()
z_vals = zv.ravel()

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
las.write("flat_ground_surface.las")