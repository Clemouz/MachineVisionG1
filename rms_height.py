"""
This script calculates the RMS height using all measurement points within a window with a given size (resolution)
instead of an interpolated grid.
"""

import laspy
import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import griddata
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt

# path leading to point cloud file
las_file = "/mnt/e/Neu/Uni/12. Semester/Machine Vision Project/Data/Machine Vision Project Data 2025 (UAV and TLS)/240829_ALS_Matrice300_Svb/240829_ALS_Matrice300_Svb_Classified.las"
# size of calculation window (resolution)
window_size = 1

# read the file
def read_laz_bounds(filename):
    las = laspy.read(filename)
    return las.x, las.y, las.z, las.classification, las.x.min(), las.y.min(), las.x.max(), las.y.max()

# Save the generated raster as "GeoTIFF"
def save_raster(filename, x, y, data):
    transform = from_bounds(x.min(), y.max(), x.max(), y.min(), data.shape[1], data.shape[0])
    with rasterio.open(filename, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1, dtype=data.dtype, transform=transform) as dst:
        dst.write(data, 1)

# Show the raster
def show_raster(filepath, title="Raster"):
    with rasterio.open(filepath) as src:
        data = src.read(1)
        plt.imshow(data, cmap='terrain')
        plt.colorbar(label='RMS Height (metres)')
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

# get point cloud data
x, y, z, classification, x_min, y_min, x_max, y_max = read_laz_bounds(las_file)
# filter out ground points
ground_points = (classification == 2)
points = np.column_stack((x[ground_points], y[ground_points], z[ground_points]))

# Build the KDTree
tree = KDTree(points[:, :2])

# Grid size
n_cols = int(np.ceil((x_max - x_min) / window_size))
n_rows = int(np.ceil((y_max - y_min) / window_size))

# Output raster array
raster = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

# Loop over grid
for row in range(n_rows):
    for col in range(n_cols):
        # Calculate window borders
        min_x = x_min + col * window_size
        max_x = min_x + window_size
        min_y = y_min + row * window_size
        max_y = min_y + window_size

        # Calculate window center and radius
        center = [(min_x + max_x) / 2, (min_y + max_y) / 2]
        radius = np.sqrt(2 * (window_size / 2) ** 2)

        # Find candidate points within the given radius with respect to the center of the window
        idxs = tree.query_ball_point(center, r=radius)
        if not idxs:
            continue  # no points in this window

        candidate_points = points[idxs]
        #print(f"number of points in radius: {len(candidate_points)}")

        # Filter exact window
        window_points = candidate_points[
            (candidate_points[:, 0] >= min_x) & (candidate_points[:, 0] < max_x) &
            (candidate_points[:, 1] >= min_y) & (candidate_points[:, 1] < max_y)
            ]
        #print(f"number of points in exact window: {len(window_points)}")

        # calculate rms height for points within the exact window and add to raster
        #if len(window_points) == 1:
        #    rms_height = window_points[0][2]
        #    print(f"rms height: {rms_height}")
        #    raster[row, col] = 0
        if len(window_points) > 1:
            z = window_points[:, 2]
            rms_height = np.sqrt(np.mean((z - z.mean()) ** 2))  # RMS height
            #print(f"rms height: {rms_height}")
            raster[row, col] = rms_height

# Prepare x, y for save function
x = np.linspace(x_min, x_max, n_cols)
y = np.linspace(y_min, y_max, n_rows)

# Save to GeoTIFF
save_raster("rms_height_map.tif", x, y, raster)
# Show the plot
show_raster("rms_height_map.tif", "RMS Height per Patch")
