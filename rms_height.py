"""
Calculating the RMS height using every measurement point instead of a grid

"""
import laspy
import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import griddata
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt


las_file = "/mnt/e/Neu/Uni/12. Semester/Machine Vision Project/Data/Machine Vision Project Data 2025 (UAV and TLS)/240829_ALS_Matrice300_Svb/240829_ALS_Matrice300_Svb_Classified.las"

def read_laz_bounds(filename):
    las = laspy.read(filename)
    return las.x, las.y, las.z, las.classification, las.x.min(), las.y.min(), las.x.max(), las.y.max()

# Save the generated raster as "GeoTIFF"
def save_raster(filename, x, y, data):
    transform = from_bounds(x.min(), y.max(), x.max(), y.min(), data.shape[1], data.shape[0])
    with rasterio.open(filename, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1, dtype=data.dtype, transform=transform) as dst:
        dst.write(data, 1)

def show_raster(filepath, title="Raster"):
    with rasterio.open(filepath) as src:
        data = src.read(1)
        plt.imshow(data, cmap='terrain')
        plt.colorbar(label='Elevation (meters)')
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

# get point cloud data
x, y, z, classification, x_min, y_min, x_max, y_max = read_laz_bounds(las_file)
ground_points = (classification == 2)
points = np.column_stack((x, y, z))

# Build the KDTree
tree = KDTree(points[:, :2])  # Only x, y if you are sliding in 2D

# Parameters
window_size = 1
# Grid size
n_cols = int(np.ceil((x_max - x_min) / window_size))
n_rows = int(np.ceil((y_max - y_min) / window_size))

# Output raster array
raster = np.full((n_rows, n_cols), np.nan, dtype=np.float32)

# Loop over grid
for row in range(n_rows):
    for col in range(n_cols):
        min_x = x_min + col * window_size
        max_x = min_x + window_size
        min_y = y_min + row * window_size
        max_y = min_y + window_size

        center = [(min_x + max_x) / 2, (min_y + max_y) / 2]
        radius = np.sqrt(2 * (window_size / 2) ** 2)

        # Find candidate points
        idxs = tree.query_ball_point(center, r=radius)
        if not idxs:
            continue  # no points in this window

        candidate_points = points[idxs]

        # Filter exact window
        window_points = candidate_points[
            (candidate_points[:, 0] >= min_x) & (candidate_points[:, 0] < max_x) &
            (candidate_points[:, 1] >= min_y) & (candidate_points[:, 1] < max_y)
            ]

        if window_points.shape[0] > 0:
            z = window_points[:, 2]
            rms_height = np.sqrt(np.mean((z - z.mean()) ** 2))  # RMS height
            raster[row, col] = rms_height

# Prepare x, y for save function
x = np.linspace(x_min, x_max, n_cols)
y = np.linspace(y_min, y_max, n_rows)

# Save to GeoTIFF
save_raster("rms_height_map.tif", x, y, raster)
show_raster("rms_height_map.tif", "RMS Height per Patch")

print(x_min)


print(points)
print(points[0])
print(x[0])