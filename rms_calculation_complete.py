"""
This file contains all approaches for calculating the rms height, creating the maps and plotting them
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
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.spatial import cKDTree  # Faster than KDTree

### CONFIG ###
#las_file = "/mnt/e/Neu/Uni/12. Semester/Machine Vision Project/Data/Machine Vision Project Data 2025 (UAV and TLS)/240829_ALS_Matrice300_Svb/240829_ALS_Matrice300_Svb_Classified.las"
las_file = "varying_point_density_slope_surface.las"
window_size = 3  # specifies the size of a window in meters (one window is one pixel in the raster and in the final map)
resolution = 1  # specifies the size of one grid cell (the point cloud height gets interpolated at the grid junctions)
n = 4  # number of nearest points (neighbours) taken into account for circle calculation

### FUNCTIONS ###
""" read the file
Args:
    filename (str): The location of the laz/ las file

Returns:
    las.x: the x values of the point cloud's points
    las.y: the y values of the point cloud's points
    las.z: the z values of the point cloud's points
    las.classification: the classification of the point cloud's points
    las.x.min(): the minimum x value
    las.x.max(): the maximum x value
    las.y.min(): the minimum y value
    las.y.max(): the maximum y value
"""
def read_laz_bounds(filename):
    las = laspy.read(filename)
    return las.x, las.y, las.z, las.classification, las.x.min(), las.y.min(), las.x.max(), las.y.max()

# Save the generated raster as "GeoTIFF"
def save_raster(filename, x, y, data):
    transform = from_bounds(x.min(), y.max(), x.max(), y.min(), data.shape[1], data.shape[0])
    with rasterio.open(filename, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1, dtype=data.dtype, transform=transform) as dst:
        dst.write(data, 1)

# Plot the given DTM map
def show_raster(filepath, title="Raster", vmin=None, vmax=None, plot_colorbar="value", info_text=None):
    with rasterio.open(filepath) as src:
        data = src.read(1)
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        plt.figure(figsize=(10, 10))
        plt.imshow(data, cmap='viridis', extent=extent, origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(label=plot_colorbar)
        plt.title(title)
        plt.xlabel("X")

        plt.ylabel("Y")
        ax = plt.gca()
        ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
        ax.xaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_major_locator(AutoLocator())

        if info_text:
            plt.gcf().text(0.02, 0.1, info_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.7))


        plt.show()

# Generate grid
def create_grid(min_x, min_y, max_x, max_y):
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

def idw_interpolate_points(x, y, z, grid_x, grid_y, power=1, max_neighbors=12):
    interpolated = np.full(grid_x.shape, np.nan)
    known_points = np.column_stack((x, y))
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    print("I Am here!")

    # Build KDTree
    tree = cKDTree(known_points)

    # Query nearest neighbors
    distances, idxs = tree.query(grid_points, k=max_neighbors, workers = -1, distance_upper_bound=4*window_size)

    # Handle case when only one neighbor is returned
    if max_neighbors == 1:
        distances = distances[:, np.newaxis]
        idxs = idxs[:, np.newaxis]

    # Compute weights
    with np.errstate(divide='ignore'):
        weights = 1.0 / np.power(distances, power)
        weights[distances == 0] = 1e12  # Assign high weight to exact matches

    # Weighted average
    z = np.append(z, np.nan)
    weighted_vals = weights * z[idxs]
    interpolated_vals = np.sum(weighted_vals, axis=1) / np.sum(weights, axis=1)

    return interpolated_vals.reshape(grid_x.shape)


# Calculate rms for specified window size and data given as grid
# dtm_array: 2d array of the dtm (extracted from the other functions)
# window_size: square size for the window
def calculate_rms(dtm_array):
    rows, cols = dtm_array.shape
    rms_map = np.full_like(dtm_array, np.nan, dtype=np.float32)

    n_of_points_per_window = int(window_size / resolution)

    for i in range(0, rows, n_of_points_per_window):
        for j in range(0, cols, n_of_points_per_window):
            window = dtm_array[i:i + n_of_points_per_window, j:j + n_of_points_per_window]

            rms = np.std(window)
            rms_map[i:i + n_of_points_per_window, j:j + n_of_points_per_window] = rms

    return rms_map

""" Uses the interpolation approach to create the rms map

1. Interpolate for discrete points in grid via nearest neighbor method globally
2. Create raster with given window size for each pixel
3. Compute RMS for all grid points in current window

"""
def interploation_approach():
    # read in the file given in the config
    x, y, z, classification, min_x, min_y, max_x, max_y = read_laz_bounds(las_file)
    # create a grid which covers the full horizontal extent of the data
    grid_x, grid_y = create_grid(min_x, min_y, max_x, max_y)
    # filter for ground points
    ground_points = (classification == 2)
    # interpolate over the height to get height values at the grid junctions
    interpolated_z = idw_interpolate_points(x[ground_points], y[ground_points], z[ground_points], grid_x, grid_y)
    # calculate the rms values
    rms_map = calculate_rms(interpolated_z)

    # save and plot the result
    save_raster("rms_height_map.tif", grid_x, grid_y, rms_map)
    info_text = "Resolution: " + str(resolution) + "\nWindow size: " + str(window_size) + "\nInterpolation method: idw (increased upper bound)"
    show_raster("rms_height_map.tif", "RMS Height", info_text=info_text)


""" Uses the real measurement approach to create the rms map

1. Filter all points which fall into current window
2. Calculate RMS for current window

"""
def real_measurements_approach():
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

    # debug: total number of points taken for calculation
    sum_window_points = 0

    # Loop over grid
    for row in tqdm(range(n_rows)):
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
            # print(f"number of points in radius: {len(candidate_points)}")

            # Filter exact window
            window_points = candidate_points[
                (candidate_points[:, 0] >= min_x) & (candidate_points[:, 0] < max_x) &
                (candidate_points[:, 1] >= min_y) & (candidate_points[:, 1] < max_y)
                ]
            # print(f"number of points in exact window: {len(window_points)}")

            sum_window_points += len(window_points)

            if len(window_points) > 0:
                z = window_points[:, 2]
                rms_height = np.sqrt(np.mean((z - z.mean()) ** 2))  # RMS height
                # print(f"rms height: {rms_height}")
                raster[row, col] = rms_height

    print(points[0])

    print(f"total number of window points: {sum_window_points}")

    # Prepare x, y for save function
    x = np.linspace(x_min, x_max, n_cols)
    y = np.linspace(y_min, y_max, n_rows)

    # Save to GeoTIFF
    save_raster("rms_height_map.tif", x, y, raster)
    # Show the plot
    show_raster("rms_height_map.tif", "RMS Height per Patch")

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
    fig, ax = plt.subplots(figsize=(20, 20))
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


""" Uses the circle approach to create a rms map

1. Iterate over all points
2. Filter nearest n neighbors for current point
3. Calculate RMS for current neighborhood

"""
def circle_approach():
    # get point cloud data
    x, y, z, classification, x_min, y_min, x_max, y_max = read_laz_bounds(las_file)
    # filter out ground points
    ground_points = (classification == 2)
    points = np.column_stack((x[ground_points], y[ground_points], z[ground_points]))

    #debug: only use every e.g. 128-th point for testing:
    points = points[::1]

    # Build the KDTree
    tree = KDTree(points[:, :2])

    results = []

    # Loop over all points
    for idx, (x, y, z) in enumerate(tqdm(points)):

        # Get k nearest neighbors
        distances, indices = tree.query([x, y], k=n)

        if len(indices) < n:
            continue  # not enough neighbors

        neighbor_z = np.append(points[indices, 2], points[idx, 2]) # take heights of neighbors and point itself
        radius = distances[-1]  # radius to the n-th neighbor
        rms_height = np.sqrt(np.mean((neighbor_z - neighbor_z.mean()) ** 2))

        results.append((x, y, z, radius, rms_height))

    # Convert to array
    results_array = np.array(results, dtype=np.float32)

    # save the points to a geo package file
    save_points("rms_points.gpkg", results_array)
    # plot points
    show_points("rms_points.gpkg")


interploation_approach()
real_measurements_approach()
circle_approach()
