import laspy
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata
from matplotlib.ticker import AutoLocator


# === CONFIG ===
las_file_1 = "RadarTower001_Classified.las"
las_file_2 = "RadarTower002_Classified.las"
resolution = 0.1  # grid resolution in meters
radar_wavelength = 0.23  # L-band radar wavelength in meters
window_size = 15  # RMS window size in pixels

# === FUNCTIONS ===

def read_laz_bounds(filename):
    las = laspy.read(filename)
    return las.x, las.y, las.z, las.classification, las.x.min(), las.y.min(), las.x.max(), las.y.max()

def create_grid(min_x, min_y, max_x, max_y, resolution=0.1):
    x = np.arange(min_x, max_x, resolution)
    y = np.arange(min_y, max_y, resolution)
    return np.meshgrid(x, y)

def interpolate_points(x, y, z, grid_x, grid_y):
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    points = np.column_stack((x, y))
    interpolated_z = griddata(points, z, grid_points, method='nearest')
    return interpolated_z.reshape(grid_x.shape)

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

def show_raster(filepath, title="Raster"):
    with rasterio.open(filepath) as src:
        data = src.read(1)
        bounds = src.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
        plt.imshow(data, cmap='terrain', extent=extent, origin='upper')
        plt.colorbar(label='Elevation (meters)')
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        ax = plt.gca()
        ax.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
        ax.xaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_major_locator(AutoLocator())

        plt.show()

# --- Suggest radar wavelength needed to achieve a specific ks target ---
def suggest_wavelength(rms_map, target_ks):
    # Given a target ks, compute the required radar wavelength per pixel
    # λ = 2π * s / ks_target
    wavelength_map = (2 * np.pi * rms_map) / target_ks
    return wavelength_map

# --- Compute ks map and classify surface roughness regions ---
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
    plt.imshow(ks_class_map, cmap=cmap, interpolation='nearest', extent=extent, origin='upper')
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

# === APPLICATION ===

# Read both LAS files
x1, y1, z1, classification1, min_x1, min_y1, max_x1, max_y1 = read_laz_bounds(las_file_1)
x2, y2, z2, classification2, min_x2, min_y2, max_x2, max_y2 = read_laz_bounds(las_file_2)

# Find common area
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
dtm_z_1 = interpolate_points(x1[ground1], y1[ground1], z1[ground1], grid_x, grid_y)
dtm_z_2 = interpolate_points(x2[ground2], y2[ground2], z2[ground2], grid_x, grid_y)

# --- Combine the DTMs ---
# Rule:
# - If both have values → average them
# - If only one has value → use that
# - If none → leave as NaN
combined_dtm = np.nanmean(np.array([dtm_z_1, dtm_z_2]), axis=0)

# Save combined DTM to GeoTIFF
save_raster("combined_dtm.tif", grid_x, grid_y, combined_dtm)

# Calculate local surface roughness (RMS)
rms_map = calculate_rms(combined_dtm, window_size)
save_raster("rms_height_map_combined.tif", grid_x, grid_y, rms_map)
show_raster("rms_height_map_combined.tif", "RMS Height per Patch (Combined)")

# Compute ks map and display classification
compute_and_show_ks_classified(rms_map, grid_x, grid_y, radar_wavelength)
