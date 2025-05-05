"""
This script calculates the RMS height using a given number of nearest measurement points within a radius (neighbours)
instead of an interpolated grid.
"""
import laspy
import numpy as np
from scipy.spatial import KDTree
from scipy.interpolate import griddata
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoLocator
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import matplotlib.cm as cm
import matplotlib.colors as colors

# path leading to point cloud file
las_file = "/mnt/e/Neu/Uni/12. Semester/Machine Vision Project/Data/Machine Vision Project Data 2025 (UAV and TLS)/240829_ALS_Matrice300_Svb/240829_ALS_Matrice300_Svb_Classified.las"
# number of nearest points (neighbours) taken into account for calculation
k = 4

# read the file
def read_laz_bounds(filename):
    las = laspy.read(filename)
    return las.x, las.y, las.z, las.classification, las.x.min(), las.y.min(), las.x.max(), las.y.max()

def save_points(filename, data):
    # Create DataFrame
    df = pd.DataFrame(data, columns=["x", "y", "z", "radius", "rms_height"])
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["x"], df["y"]), crs="EPSG:32634")  # WGS 84 / UTM zone 34N
    # Save as GeoPackage or Shapefile
    gdf.to_file(filename, driver="GPKG")  # or driver="ESRI Shapefile"

def show_points(filename):
    # Load your GeoPackage
    gdf = gpd.read_file(filename)

    # Create circles
    circles = []
    rms_values = gdf["rms_height"].values
    radii = gdf["radius"].values

    # Normalize RMS values to a color scale
    norm = colors.Normalize(vmin=np.nanmin(rms_values), vmax=np.nanmax(rms_values))
    cmap = cm.viridis  # or 'plasma', 'magma', etc.
    face_colors = [cmap(norm(r)) for r in rms_values]

    #print(face_colors)

    for (x, y, r) in tqdm(zip(gdf.geometry.x, gdf.geometry.y, radii)):
        circle = Circle((x, y), r)
        circles.append(circle)

    #print(circles)

    # Plot
    fig, ax = plt.subplots(figsize=(50, 50))
    collection = PatchCollection(circles, facecolor=face_colors, edgecolor='black', linewidth=0.5, alpha=0.5)
    ax.add_collection(collection)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("RMS Height")

    ax.set_aspect('equal')
    ax.set_title("RMS Height Circles")
    ax.autoscale()
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# get point cloud data
x, y, z, classification, x_min, y_min, x_max, y_max = read_laz_bounds(las_file)
# filter out ground points
ground_points = (classification == 2)
points = np.column_stack((x[ground_points], y[ground_points], z[ground_points]))

#debug: only use every n-th point for testing:
points = points[::128]

# Build the KDTree
tree = KDTree(points[:, :2])

#debug: total number of points taken for calculation
#sum_window_points = 0

results = []

# Loop over all points
for idx, (x, y, z) in enumerate(tqdm(points)):

    # Get k nearest neighbors
    distances, indices = tree.query([x, y], k=k)

    if len(indices) < k:
        continue  # not enough neighbors

    neighbor_z = np.append(points[indices, 2], points[idx, 2]) # take heights of neighbors and point itself
    radius = distances[-1]  # radius to the k-th neighbor
    rms_height = np.sqrt(np.mean((neighbor_z - neighbor_z.mean()) ** 2))

    results.append((x, y, z, radius, rms_height))

#print(f"total number of window points: {sum_window_points}")

# Convert to array
results_array = np.array(results, dtype=np.float32)

# save the points to a geo package file
save_points("rms_points.gpkg", results_array)
# plot points
show_points("rms_points.gpkg")

