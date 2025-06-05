"""
Terrain Surface Roughness Estimation and Scattering Suitability Analysis

Author: Group 1 — Machine Vision Project Course, Spring 2025
Course: 5DV190 – Project Course in Machine Vision, Umeå University

Description:
This script processes one or more las/laz files to compute terrain surface roughness using RMS height, correlation length
and evaluate radar scattering model suitability based on the surface roughness parameter.

Inputs:
- One or more classified .las files (ASPRS LAS format)
- Radar wavelength in meters
- User-specified or automatic resolution and window size
- Max_neighbors for IDW interpolation (used in RMS)

Outputs:
- Correlation length map (GeoTIFF file and plotted)
- RMS height map (GeoTIFF file and plotted)
- Classified kₛ plot indicating valid scattering models (SPM/IEM/GOM)

Usage:
- User is prompted via console for input mode and parameters
- All plots show georeferenced axes with colorbars and terrain overlays

Dependencies:
- laspy, numpy, matplotlib, rasterio, tqdm, scipy

"""



import laspy
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import AutoLocator
from scipy.signal import correlate2d
import math
from scipy.spatial import cKDTree  


# === CONFIG ===
#las_file_1 = "RadarTower001_Classified.las"
#las_file_2 = "240829_ALS_Matrice300_Svb_Classified.las"
resolution = 0.1  # defult grid resolution in meters
radar_wavelength = 0.23  # defult radar wavelength in meters
window_size_p = 15  # defult window size in pixels
window_size_m = 1.5 # defult window size in meters
max_neighbors = 12 # defult number of neighbors used for idw_interpolate_points

# === FUNCTIONS ===

# Function: read_laz_bounds
# Description:
#   Reads a LAS point cloud file and extracts x, y, z coordinates,
#   classification labels, and the spatial bounds of the data.
# Input:
#   - filename: Path to a .las file (string)
# Output:
#   - x, y, z: Arrays of point coordinates
#   - classification: Array of classification codes
#   - min_x, min_y, max_x, max_y: Bounding box of the point cloud
def read_laz_bounds(filename):
    las = laspy.read(filename)
    return las.x, las.y, las.z, las.classification, las.x.min(), las.y.min(), las.x.max(), las.y.max()

# Function: create_grid
# Description:
#   Generates a uniform 2D grid covering the bounding box with the specified
#   spatial resolution. Used for rasterizing interpolated values.
# Input:
#   - min_x, min_y: Minimum coordinates of the area (floats)
#   - resolution: Grid cell size in meters (default: 0.1)
# Output:
#   - grid_x, grid_y: 2D meshgrids of x and y coordinates
def create_grid(min_x, min_y, max_x, max_y, resolution=0.1):
    x = np.arange(min_x, max_x, resolution)
    y = np.arange(min_y, max_y, resolution)
    return np.meshgrid(x, y)

# Description:
#   Performs Inverse Distance Weighting (IDW) interpolation on scattered
#   LiDAR ground points to estimate elevation (z-values) on a regular grid.
#   For each target grid cell (x, y), the function finds the nearest known 
#   points using a KDTree and computes a weighted average of their z-values 
#   based on inverse distance to the target point.
# Inputs:
#   - x, y, z          : 1D arrays of known ground point coordinates and elevations
#   - grid_x, grid_y   : 2D arrays (meshgrids) of target x/y coordinates for interpolation
#   - power            : Weighting power for IDW (default = 1); higher values give more influence to closer points
#   - max_neighbors    : Maximum number of nearest neighbors to consider for each grid point (user-defined)
# Output:
#   - A 2D array of interpolated z-values
def idw_interpolate_points(x, y, z, grid_x, grid_y, power=1, max_neighbors=12):
    interpolated = np.full(grid_x.shape, np.nan)
    known_points = np.column_stack((x, y))
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))

    # Build KDTree
    tree = cKDTree(known_points)

    # Query nearest neighbors
    distances, idxs = tree.query(grid_points, k=max_neighbors, workers = -1, distance_upper_bound=window_size_p)

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


# Function: calculate_rms
# Description:
#   Computes the RMS (root mean square) height of the terrain surface using
#   a sliding window approach. For each square patch of the input DTM array,
#   it calculates the standard deviation
#   and fills a new raster (rms_map) with the same value across that patch.
#   This results in a blocky map where each window shares a uniform RMS value.
# Inputs:
#   - dtm_array      : 2D array representing the digital terrain model (interpolated surface)
#   - window_size_p  : Size of the window in pixels used to compute local RMS height
# Output:
#   - rms_map        : 2D array containing RMS height values
def calculate_rms(dtm_array, window_size_p):
    rows, cols = dtm_array.shape
    rms_map = np.full_like(dtm_array, np.nan, dtype=np.float32)

    for i in range(0, rows, window_size_p):
        for j in range(0, cols, window_size_p):
            window = dtm_array[i:i+window_size_p, j:j+window_size_p]
            rms = np.std(window)
            rms_map[i:i+window_size_p, j:j+window_size_p] = rms

    return rms_map

# Function: save_raster
# Description:
#   Saves a 2D NumPy array as a GeoTIFF raster file using spatial referencing
#   derived from the input coordinate bounds. This is used to export processed
#   data (e.g., RMS maps, correlation maps) into a format compatible with
#   GIS tools such as QGIS or ArcGIS.
# Inputs:
#   - filename : Output path and filename for the raster (string)
#   - x, y     : 1D coordinate arrays defining the spatial extent (must match the grid used)
#   - data     : 2D NumPy array of the raster data (e.g., elevation, RMS, etc.)
# Output:
#   - Writes a GeoTIFF (.tif) file to disk with georeferenced bounds
def save_raster(filename, x, y, data):
    transform = from_bounds(x.min(), y.max(), x.max(), y.min(), data.shape[1], data.shape[0])
    with rasterio.open(filename, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1, dtype=data.dtype, transform=transform) as dst:
        dst.write(data, 1)

# Function: show_raster
# Description:
#   Displays a georeferenced raster image (GeoTIFF) using matplotlib. 
#   It includes a colorbar, coordinate axis labels, optional gridlines, 
#   and additional information text embedded on the figure.
# Inputs:
#   - filepath       : Path to the GeoTIFF file (string)
#   - title          : Title of the plot (string, default: "Raster")
#   - plot_colorbar  : Label for the colorbar (string, default: "value")
#   - info_text      : Optional string displayed on the side of the plot, 
#                      useful for including metadata or file info (default: None)
# Output:
#   - Opens a matplotlib plot window displaying the raster with coordinates
#     and additional annotations.
def show_raster(filepath, title="Raster", plot_colorbar="value", info_text=None):
    with rasterio.open(filepath) as src:
        data = src.read(1)
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        plt.imshow(data, cmap='viridis', extent=extent, origin='lower')
        plt.colorbar(label=plot_colorbar)
        plt.title(title)
        plt.xlabel("X")

        plt.ylabel("Y")
        ax = plt.gca()
        ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
        ax.xaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_major_locator(AutoLocator())

        if info_text:
            plt.gcf().text(0.02, 0.5, info_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.7))


        plt.show()

# Function: suggest_wavelength
# Description:
#   Calculates the required radar wavelength (λ) at each grid cell to achieve
#   a specified surface roughness parameter (k_s). This is useful for evaluating
#   which radar wavelengths are suitable for different surface roughness levels.
# Inputs:
#   - rms_map    : 2D NumPy array containing RMS height values (s) per grid cell
#   - target_ks  : Desired surface roughness parameter k_s (float)
#                  e.g., 0.3 for SPM, 1.5 for IEM, or 3.5 for GOM
# Output:
#   - wavelength_map : 2D NumPy array of computed radar wavelengths λ,
#                      using the formula: λ = (2π * s) / k_s
def suggest_wavelength(rms_map, target_ks):
    wavelength_map = (2 * np.pi * rms_map) / target_ks
    return wavelength_map

# Function: compute_and_show_ks_classified
# Description:
#   Computes the dimensionless surface roughness parameter k_s for each grid cell
#   using the RMS height and a given radar wavelength. It then classifies the 
#   surface into three types (Smooth, Moderate, Rough) based on k_s thresholds
#   and visualizes the classification with color-coded mapping.
# Inputs:
#   - rms_map          : 2D NumPy array of RMS height values (s) in meters
#   - grid_x, grid_y   : 2D meshgrids defining the spatial extent (used for plotting)
#   - radar_wavelength : Radar wavelength (λ) in meters (default = 0.23 m)
# Output:
#   - Displays a matplotlib figure showing surface roughness categories using:
#       - Blue for Smooth (SPM model)
#       - Green for Moderate (IEM model)
#       - Red for Rough (GOM model)
#     Also prints wavelength suggestions for each model based on avg RMS.
def compute_and_show_ks_classified(rms_map, grid_x, grid_y, radar_wavelength=0.23):
    # Convert RMS height to ks using radar wavelength: ks = s * 2π / λ
    k = 2 * np.pi / radar_wavelength
    ks_map = rms_map * k

    # Classify into categories: 0=smooth, 1=moderate, 2=rough
    ks_class_map = np.full_like(ks_map, np.nan)
    ks_class_map[(ks_map < 0.3)] = 0  # Smooth (SPM)
    ks_class_map[(ks_map >= 0.3) & (ks_map < 3)] = 1  # Moderate (IEM)
    ks_class_map[(ks_map >= 3)] = 2  # Rough (GOM)

    # Visualize
    cmap = ListedColormap(['blue', 'green', 'red'])
    plt.figure(figsize=(12, 10))
    bounds = [grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()]
    extent = [bounds[0], bounds[1], bounds[2], bounds[3]]
    plt.imshow(ks_class_map, cmap=cmap, interpolation='nearest', extent=extent, origin='lower')
    plt.colorbar(ticks=[0.5, 1.5, 2.5], label="Surface Type")
    plt.clim(0, 3)
    plt.title("Surface Roughness Classification Based on k_s")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Legend for surface model types
    labels = ["Smooth (SPM)", "Moderate (IEM)", "Rough (GOM)"]
    colors = ['blue', 'green', 'red']
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors]
    plt.legend(handles, labels, loc='lower right', title="Surface Type")

    # Suggest radar wavelengths for each model type based on average RMS
    avg_rms = np.nanmean(rms_map)
    wavelength_for_spm = (2 * np.pi * avg_rms) / 0.2
    wavelength_for_iem = (2 * np.pi * avg_rms) / 1.5
    wavelength_for_gom = (2 * np.pi * avg_rms) / 3.5

    suggestion_text = f"Suggested λ:\nSPM: {wavelength_for_spm:.2f} m\nIEM: {wavelength_for_iem:.2f} m\nGOM: {wavelength_for_gom:.2f} m"
    plt.gcf().text(0.02, 0.5, suggestion_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    ax = plt.gca()
    ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
    ax.xaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_major_locator(AutoLocator())
    plt.show()

# Function: calculate_correlation_length
# Description:
#   Computes the spatial correlation length of surface heights using the 
#   2D autocorrelation function (ACF) for each window of the input DTM (Digital 
#   Terrain Model). The correlation length is defined as the distance at which 
#   the ACF drops below 1/e (~0.3679) of its maximum (central) value. 
# Inputs:
#   - dtm_array      : 2D NumPy array representing the digital terrain model
#   - window_size_p  : Size (in pixels) of the moving window used to compute
#                      correlation length locally
# Output:
#   - corr_map       : 2D NumPy array of same shape as dtm_array, containing 
#                      calculated correlation length for each window
def calculate_correlation_length(dtm_array, window_size_p):
    rows, cols = dtm_array.shape
    corr_map = np.full_like(dtm_array, np.nan, dtype=np.float32)
    acf_mean = 0
    for i in range(0, rows, window_size_p):
        for j in range(0, cols, window_size_p):
            window = dtm_array[i:i+window_size_p, j:j+window_size_p]

            if np.isnan(window).all():
                continue

            window = window - np.nanmean(window)
            window = np.nan_to_num(window)

            acf = correlate2d(window, window, mode='full', boundary='symm')
            center = (acf.shape[0] // 2, acf.shape[1] // 2)
            center_val = acf[center]

            if center_val == 0 or np.isnan(center_val):
                continue

            acf = acf / center_val
            acf_mean += np.nanmean(acf)
            y_idxs, x_idxs = np.indices(acf.shape)
            r = np.sqrt((x_idxs - center[1])**2 + (y_idxs - center[0])**2).astype(np.int32)
            r = r.astype(np.int32)
            max_radius = r.max()
            radial_acf = np.zeros(max_radius + 1)
            counts = np.zeros(max_radius + 1)

            for rad in range(max_radius + 1):
                mask = r == rad
                values = acf[mask]
                if values.size > 0:
                    radial_acf[rad] = np.mean(values)
                    counts[rad] = values.size

            threshold = 1 / math.e
            below = np.where(radial_acf < threshold)[0]
            if below.size > 0:
                corr_length = below[0]
            else:
                corr_length = max_radius

            corr_map[i:i+window_size_p, j:j+window_size_p] = corr_length
    #print("Mean ACF: ", acf_mean/(rows*cols))
    return corr_map

# Function: calc_corr_length_and_rms
# Description:
#   Computes both the spatial correlation length and the RMS height of the
#   terrain surface for each window in a Digital Terrain Model (DTM) array.
# Inputs:
#   - dtm_array      : 2D NumPy array containing the gridded DTM values
#   - window_size_p  : Integer specifying the size of the square window (in pixels)
#                      over which the calculations are performed
# Outputs:
#   - corr_map       : 2D NumPy array of the same shape as dtm_array, containing 
#                      the correlation length values for each window
#   - rms_map        : 2D NumPy array of the same shape as dtm_array, containing 
#                      RMS surface height values for each window
def calc_corr_length_and_rms(dtm_array, window_size_p):
    rows, cols = dtm_array.shape
    corr_map = np.full_like(dtm_array, np.nan, dtype=np.float32)
    rms_map = np.full_like(dtm_array, np.nan, dtype=np.float32)
    acf_mean = 0
    for i in range(0, rows, window_size_p):
        for j in range(0, cols, window_size_p):
            window = dtm_array[i:i+window_size_p, j:j+window_size_p]
            rms = np.std(window)
            rms_map[i:i+window_size_p, j:j+window_size_p] = rms
            if np.isnan(window).all():
                continue

            window = window - np.nanmean(window)
            window = np.nan_to_num(window)

            acf = correlate2d(window, window, mode='full', boundary='symm')
            center = (acf.shape[0] // 2, acf.shape[1] // 2)
            center_val = acf[center]

            if center_val == 0 or np.isnan(center_val):
                continue

            acf = acf / center_val
            acf_mean += np.nanmean(acf)
            y_idxs, x_idxs = np.indices(acf.shape)
            r = np.sqrt((x_idxs - center[1])**2 + (y_idxs - center[0])**2).astype(np.int32)
            r = r.astype(np.int32)
            max_radius = r.max()
            radial_acf = np.zeros(max_radius + 1)
            counts = np.zeros(max_radius + 1)

            for rad in range(max_radius + 1):
                mask = r == rad
                values = acf[mask]
                if values.size > 0:
                    radial_acf[rad] = np.mean(values)
                    counts[rad] = values.size

            threshold = 1 / math.e
            below = np.where(radial_acf < threshold)[0]
            if below.size > 0:
                corr_length = below[0]
            else:
                corr_length = max_radius

            corr_map[i:i+window_size_p, j:j+window_size_p] = corr_length
    #print("Mean ACF: ", acf_mean/(rows*cols))
    return corr_map, rms_map


# === USER INPUT SECTION ===
# This block handles all user inputs required for running the processing pipeline.
# Users can input one or more LAS file paths, choose between automatic or manual
# configuration modes, and select which plots to generate (RMS, k_s, correlation length).
# Based on the selected mode, parameters like resolution, window size, and interpolation
# neighbors are calculated or collected.
try:
    input_str = input("Enter one or more LAS file paths, separated by commas:\n").strip()
    las_files = [f.strip() for f in input_str.split(",") if f.strip()]
    if not las_files:
        print("No valid LAS files provided. Exiting.")
        exit(1)

    mode = input("Choose input mode:\n1 - Use automatic wavelength-based settings\n2 - Use manual custom input\nEnter mode (1 or 2): ").strip()

    if mode == "1":
        radar_wavelength = float(input("Enter radar wavelength in meters (e.g. 0.23): "))
        resolution = 1 * radar_wavelength
        window_size_p = int(10 * radar_wavelength / resolution)  # in pixels
        window_size_m = round((10 * radar_wavelength),3) # in meters

    elif mode == "2":
        radar_wavelength = float(input("Enter radar wavelength in meters (e.g. 0.23): "))
        resolution = float(input("Enter grid resolution in meters (e.g. 0.1): "))
        window_size_m = float(input("Enter window size in meters (e.g. 1.5): "))
        if (window_size_m < resolution): 
            raise ValueError("The window size must be bigger than the resolution")
        window_size_p = int(window_size_m / resolution)  # convert to pixels

    else:
        raise ValueError("Invalid mode selection.")

    plot_choice = input("Select processing:\n1 - RMS and ks maps\n2 - Correlation length\n3 - All\nEnter selection (1, 2, or 3): ").strip()
    
    if plot_choice not in ("1", "2", "3"):
        raise ValueError("Invalid plot selection.")

    if plot_choice in ("1", "3"):
        max_neighbors_input = int(input("Enter max_neighbors for the IDW interpolation (minimum 1): "))
        if max_neighbors_input < 1:
            raise ValueError("max_neighbors must be at least 1.")
        else:
            max_neighbors = max_neighbors_input

except ValueError as e:
    print(f"Error: {e}")
    exit(1)

# === END USER INPUT SECTION ===

# === APPLICATION ===
# This section performs the full terrain roughness analysis using the input LAS files.
# Main steps:
# 1. Load and parse all provided LAS files.
# 2. Determine the total area to cover based on combined spatial extents.
# 3. Create a common grid that all point clouds will be interpolated onto.
# 4. Filter ground-classified points (class == 2) from each file.
# 5. Interpolate the elevation values onto the grid using IDW.
# 6. Merge the interpolated elevation maps into one composite DTM.
# 7. Based on the user’s selection:
#    - Compute and save RMS height map.
#    - Compute and visualize k_s surface roughness classification.
#    - Compute and save correlation length map.
#    - Optionally compute all the above.
# 8. Add metadata to plots: resolution, wavelength, window size, etc.


# Read all LAS files
all_dtms = []
file_info_lines = []
min_x, min_y = float("inf"), float("inf")
max_x, max_y = float("-inf"), float("-inf")

for las_file in las_files:
    x, y, z, classification, minx, miny, maxx, maxy = read_laz_bounds(las_file)
    min_x = min(min_x, minx)
    min_y = min(min_y, miny)
    max_x = max(max_x, maxx)
    max_y = max(max_y, maxy)
    # Filter only ground-classified points
    ground = classification == 2
    all_dtms.append((x[ground], y[ground], z[ground]))
    file_info_lines.append(f"File: {las_file}")

# Create common grid for all point clouds
grid_x, grid_y = create_grid(min_x, min_y, max_x, max_y, resolution)

# Interpolate all scans onto the same grid
interpolated_dtms = [
    idw_interpolate_points(x, y, z, grid_x, grid_y, max_neighbors=max_neighbors)
    for (x, y, z) in all_dtms
]
# --- Combine the DTMs ---
# Rule:
# - If all have values → average them
# - If only one has value → use that
# - If none → leave as NaN
stack = np.array(interpolated_dtms)  
valid_count = np.sum(~np.isnan(stack), axis=0)
combined_dtm = np.full_like(stack[0], np.nan)
mask = valid_count > 0
combined_dtm[mask] = np.nanmean(stack[:, mask], axis=0)


# Save combined DTM to GeoTIFF
save_raster("combined_dtm.tif", grid_x, grid_y, combined_dtm)

file_info_text = "Used files:\n" + "\n".join(file_info_lines)
# Info string for RMS-related plots (includes max_neighbors)
info_str_rms = f"""{file_info_text}
Resolution: {resolution} m
Wavelength: {radar_wavelength} m
max_neighbors: {max_neighbors} 
Window size: {window_size_m} m"""

# Info string for correlation plot (excludes max_neighbors)
info_str_corr = f"""{file_info_text}
Resolution: {resolution} m
Wavelength: {radar_wavelength} m
Window size: {window_size_m} m"""

# Conditional logic for plot generation
if plot_choice == "1":
    # Calculate local surface roughness (RMS)
    rms_map = calculate_rms(combined_dtm, window_size_p)
    save_raster("rms_height_map_combined.tif", grid_x, grid_y, rms_map)
    show_raster("rms_height_map_combined.tif", title="RMS Height per Patch (Combined)",plot_colorbar="RMS higths in meters", info_text=info_str_rms)
    # Compute ks map and display classification
    compute_and_show_ks_classified(rms_map, grid_x, grid_y, radar_wavelength)

if plot_choice == "2":
    corr_map = calculate_correlation_length(combined_dtm, window_size_p)
    save_raster("correlation_length_map.tif", grid_x, grid_y, corr_map)
    show_raster("correlation_length_map.tif", title="Correlation Length per Patch",plot_colorbar=" correlation length in meters", info_text=info_str_corr)

if plot_choice == "3":
    corr_map, rms_map = calc_corr_length_and_rms(combined_dtm, window_size_p)
    save_raster("correlation_length_map.tif", grid_x, grid_y, corr_map)
    save_raster("rms_height_map_combined.tif", grid_x, grid_y, rms_map)
    show_raster("correlation_length_map.tif", title="Correlation Length per Patch",plot_colorbar=" correlation length in meters", info_text=info_str_corr)
    show_raster("rms_height_map_combined.tif", title="RMS Height per Patch (Combined)",plot_colorbar="RMS higths in meters", info_text=info_str_rms)
    compute_and_show_ks_classified(rms_map, grid_x, grid_y, radar_wavelength)