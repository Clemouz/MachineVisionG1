import laspy
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata
from matplotlib.ticker import AutoLocator
from scipy.signal import correlate2d
import math
from scipy.spatial import ckdtree
from scipy.spatial import cKDTree  # Faster than KDTree


# === CONFIG ===
las_file_1 = "RadarTower002_Classified.las"
las_file_2 = "RadarTower001_Classified.las"
resolution = 0.1  # grid resolution in meters
radar_wavelength = 0.23  # L-band radar wavelength in meters
window_size = 15  # RMS window size in pixels
max_neighbors = 12 #the number of neighbors used for idw_interpolate_points

# === FUNCTIONS ===

def read_laz_bounds(filename):
    las = laspy.read(filename)
    return las.x, las.y, las.z, las.classification, las.x.min(), las.y.min(), las.x.max(), las.y.max()

def create_grid(min_x, min_y, max_x, max_y, resolution=0.1):
    x = np.arange(min_x, max_x, resolution)
    y = np.arange(min_y, max_y, resolution)
    return np.meshgrid(x, y)


def idw_interpolate_points(x, y, z, grid_x, grid_y, power=1, max_neighbors=12):
    interpolated = np.full(grid_x.shape, np.nan)
    known_points = np.column_stack((x, y))
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))


    # Build KDTree
    tree = cKDTree(known_points)

    # Query nearest neighbors
    distances, idxs = tree.query(grid_points, k=max_neighbors, workers = -1, distance_upper_bound=window_size)

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



def calculate_rms(dtm_array, window_size):
    rows, cols = dtm_array.shape
    rms_map = np.full_like(dtm_array, np.nan, dtype=np.float32)

    for i in range(0, rows, window_size):
        for j in range(0, cols, window_size):
            window = dtm_array[i:i+window_size, j:j+window_size]
            rms = np.std(window)
            rms_map[i:i+window_size, j:j+window_size] = rms

    return rms_map

def save_raster(filename, x, y, data):
    transform = from_bounds(x.min(), y.max(), x.max(), y.min(), data.shape[1], data.shape[0])
    with rasterio.open(filename, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1, dtype=data.dtype, transform=transform) as dst:
        dst.write(data, 1)

def show_raster(filepath, title="Raster", vmin=None, vmax=None, plot_colorbar="value", info_text=None):
    with rasterio.open(filepath) as src:
        data = src.read(1)
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        plt.imshow(data, cmap='terrain', extent=extent, origin='lower', vmin=vmin, vmax=vmax)
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

def suggest_wavelength(rms_map, target_ks):
    # Given a target ks, compute the required radar wavelength per pixel
    # λ = 2π * s / ks_target
    wavelength_map = (2 * np.pi * rms_map) / target_ks
    return wavelength_map

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
    wavelength_for_spm = (2 * np.pi * avg_rms) / 0.3
    wavelength_for_iem = (2 * np.pi * avg_rms) / 1.5
    wavelength_for_gom = (2 * np.pi * avg_rms) / 3.5

    suggestion_text = f"Suggested λ:\nSPM: {wavelength_for_spm:.2f} m\nIEM: {wavelength_for_iem:.2f} m\nGOM: {wavelength_for_gom:.2f} m"
    plt.gcf().text(0.02, 0.5, suggestion_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
    ax = plt.gca()
    ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
    ax.xaxis.set_major_locator(AutoLocator())
    ax.yaxis.set_major_locator(AutoLocator())
    plt.show()

def calculate_correlation_length(dtm_array, window_size):
    rows, cols = dtm_array.shape
    corr_map = np.full_like(dtm_array, np.nan, dtype=np.float32)
    acf_mean = 0
    for i in range(0, rows, window_size):
        for j in range(0, cols, window_size):
            window = dtm_array[i:i+window_size, j:j+window_size]

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

            corr_map[i:i+window_size, j:j+window_size] = corr_length
    print("Mean ACF: ", acf_mean/(rows*cols))
    return corr_map

# === USER INPUT SECTION ===
try:
    radar_wavelength = float(input("Enter radar wavelength in meters (e.g. 0.23): "))
    plot_choice = input("Enter 1 for RMS+ks, 2 for Correlation, 3 for All: ").strip()

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

# Read both LAS files
x1, y1, z1, classification1, min_x1, min_y1, max_x1, max_y1 = read_laz_bounds(las_file_1)
x2, y2, z2, classification2, min_x2, min_y2, max_x2, max_y2 = read_laz_bounds(las_file_2)

min_x = min(min_x1, min_x2)
min_y = min(min_y1, min_y2)
max_x = max(max_x1, max_x2)
max_y = max(max_y1, max_y2)

# Create common grid for both point clouds
grid_x, grid_y = create_grid(min_x, min_y, max_x, max_y, resolution)

# Filter only ground-classified points
ground1 = (classification1 == 2)
ground2 = (classification2 == 2)

# Interpolate both scans onto the same grid
dtm_z_1 = idw_interpolate_points(x1[ground1], y1[ground1], z1[ground1], grid_x, grid_y)
dtm_z_2 = idw_interpolate_points(x2[ground2], y2[ground2], z2[ground2], grid_x, grid_y)

# --- Combine the DTMs ---
# Rule:
# - If both have values → average them
# - If only one has value → use that
# - If none → leave as NaN
combined_dtm = np.nanmean(np.array([dtm_z_1, dtm_z_2]), axis=0)

# Save combined DTM to GeoTIFF
save_raster("combined_dtm.tif", grid_x, grid_y, combined_dtm)


# Info string for RMS-related plots (includes max_neighbors)
info_str_rms = f"""File 1: {las_file_1}
File 2: {las_file_2}
Resolution: {resolution} m
Wavelength: {radar_wavelength} m
max_neighbors: {max_neighbors} 
Window size: {window_size} px"""

# Info string for correlation plot (excludes max_neighbors)
info_str_corr = f"""File 1: {las_file_1}
File 2: {las_file_2}
Resolution: {resolution} m
Wavelength: {radar_wavelength} m
Window size: {window_size} px"""



# Conditional logic for plot generation
if plot_choice in ("1", "3"):
    # Calculate local surface roughness (RMS)
    rms_map = calculate_rms(combined_dtm, window_size)
    save_raster("rms_height_map_combined.tif", grid_x, grid_y, rms_map)
    show_raster("rms_height_map_combined.tif", title="RMS Height per Patch (Combined)", vmin=0, vmax=2,plot_colorbar="RMS higths in meters", info_text=info_str_rms)

    
    # Compute ks map and display classification
    compute_and_show_ks_classified(rms_map, grid_x, grid_y, radar_wavelength)

if plot_choice in ("2", "3"):
    corr_map = calculate_correlation_length(combined_dtm, window_size)
    save_raster("correlation_length_map.tif", grid_x, grid_y, corr_map)
    show_raster("correlation_length_map.tif", title="Correlation Length per Patch", vmin=0, vmax=12,plot_colorbar=" correlation length in meters", info_text=info_str_corr)

