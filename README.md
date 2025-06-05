# MachineVisionG1  
## Terrain Surface Roughness Estimator

### Overview

This project was developed as part of the **5DV190: Project Course in Machine Vision** at Umeå University, Spring 2025. It processes one or more `.las` or `.laz` LiDAR files to compute surface roughness metrics and assess radar backscatter suitability using geospatial analysis techniques.

### Output Includes

- **RMS Height Map**
- **Correlation Length Map**
- **Surface Roughness Classification Map** (based on SPM/IEM/GOM scattering models)
- **Suggested radar wavelengths** per region for each model type

---

### Features

- Accepts **multiple point cloud files** for the same area
- Performs **Inverse Distance Weighting (IDW)** interpolation
- Outputs **GeoTIFF raster files** with correct coordinate referencing
- Raster plots include **gridlines**, **color legends**, and **metadata overlays**
- User-selectable **input modes** and **processing types**
- Calculates RMS and correlation length over **local window patches**
- Suggests radar wavelengths suitable for **SPM, IEM, and GOM** models

---

### How to Use

#### 1. Clone the Repository

```bash
git clone https://github.com/Clemouz/MachineVisionG1.git
```

#### 2. Install Dependencies

Make sure the following Python packages are installed:

- `laspy`
- `numpy`
- `matplotlib`
- `rasterio`
- `tqdm`
- `scipy`

You can install them using:

```bash
pip install laspy numpy matplotlib rasterio tqdm scipy
```

---

#### 3. Run the Script

```bash
python creatGrid2_x_number_of_files.py
```

---

#### 4. Follow the Prompts

The script will prompt you to:

- Enter one or more **LAS file paths** (comma-separated)
- Choose **input mode**:
  - **1 - Automatic**: `resolution = λ`, `window size = 10λ`
  - **2 - Manual**: You specify resolution and window size
- Provide **radar wavelength**
- Provide **max_neighbors** (used for IDW interpolation)
- Select **processing type**:
  - `1` – RMS and k_s maps
  - `2` – Correlation length
  - `3` – All maps

---

### Output Files

#### GeoTIFFs:

- `rms_height_map_combined.tif`
- `correlation_length_map.tif`

#### Plots:

- **RMS Height Map**
- **Correlation Length Map**
- **Surface Roughness Classification Map**  
  (color-coded and annotated with scattering model type and suggested wavelengths)

---
