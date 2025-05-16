import numpy as np
import laspy

# Grid points
x = np.arange(0, 100, 1)
y = np.arange(0, 100, 1)
xv, yv = np.meshgrid(x, y)
x_vals = xv.ravel()
y_vals = yv.ravel()
slope_factor = 0.02

# Define bumps and dips: (center_x, center_y, amplitude, sigma)
features = [
    (30, 30, +5, 10),   # Bump
    (70, 30, -4, 8),    # Dip
    (50, 70, +3, 12),   # Bump
    (80, 80, -2.5, 10), # Dip
    (20, 80, +4, 8)     # Bump
]

# Compute elevation as sum of bumps/dips
z_vals = (x_vals + y_vals) * slope_factor
for cx, cy, amp, sigma in features:
    dist_sq = (x_vals - cx)**2 + (y_vals - cy)**2
    z_vals += amp * np.exp(-dist_sq / (2 * sigma**2))
z_vals = (x_vals + y_vals) * slope_factor

# Density centers
centers = [(30, 30), (50, 30), (70, 70)]
radius = 25  # max distance at which density still has influence
falloff_exponent = 3.0  # controls curve sharpness
min_prob = 1
max_prob = 1

# Initialize probability map
prob = np.zeros_like(x_vals, dtype=np.float32)

# Apply flattened falloff from each center
for cx, cy in centers:
    dist = np.sqrt((x_vals - cx)**2 + (y_vals - cy)**2)
    local_prob = 1 - (dist / radius)**falloff_exponent
    local_prob = np.clip(local_prob, 0, 1)
    prob += local_prob

# Normalize, then apply thresholds
prob = prob / prob.max()
prob = np.clip(prob, min_prob, max_prob)

# Random sampling using the probability map
mask = np.random.rand(x_vals.shape[0]) < prob
x_vals, y_vals, z_vals = x_vals[mask], y_vals[mask], z_vals[mask]

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
las.write("slope_surface.las")