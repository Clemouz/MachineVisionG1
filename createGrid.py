import laspy
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

las_file = "Points.laz"

# Read LAS file and get bounds
def read_laz_bounds(filename):
    las = laspy.read(filename)
    return las.x, las.y, las.z, las.classification, las.x.min(), las.y.min(), las.x.max(), las.y.max()

# Generate grid
def create_grid(min_x, min_y, max_x, max_y, resolution=0.1):
    x = np.arange(min_x, max_x, resolution)
    y = np.arange(min_y, max_y, resolution)
    return np.meshgrid(x, y)

# Fill in the grid with the ground points, using library: griddata
# Fill in a blank grid over the map where some spots don't have known height values, using the known height values at the marked points (nearest neighbour).
# x: x-coordinate
# y: y-coordinate
# z: height
def interpolate_points(x, y, z, grid_x, grid_y):
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    points = np.column_stack((x, y))
    interpolated_z = griddata(points, z, grid_points, method='nearest')
    return interpolated_z.reshape(grid_x.shape)

# Save the generated raster as "GeoTIFF"
def save_raster(filename, x, y, data):
    transform = from_bounds(x.min(), y.max(), x.max(), y.min(), data.shape[1], data.shape[0])
    with rasterio.open(filename, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1, dtype=data.dtype, transform=transform) as dst:
        dst.write(data, 1)

# Show DTM map
def show_raster(filepath, title="Raster"):
    with rasterio.open(filepath) as src:
        data = src.read(1)
        plt.imshow(data, cmap='terrain')
        plt.colorbar(label='Elevation (meters)')
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()




#Application
x, y, z, classification, min_x, min_y, max_x, max_y = read_laz_bounds(las_file)
grid_x, grid_y = create_grid(min_x, min_y, max_x, max_y, resolution=0.1)
ground_points = (classification == 2)
dtm_z = interpolate_points(x[ground_points], y[ground_points], z[ground_points], grid_x, grid_y)
save_raster("dtm.tif", grid_x, grid_y, dtm_z)
show_raster("dtm.tif", "Digital Terrain Model")
