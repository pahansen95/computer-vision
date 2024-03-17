import cv2
import numpy as np
import os
import sys
import orjson
from pathlib import Path
import matplotlib.pyplot as plt
from loguru import logger
import math

logger.remove()
# logger.add(sys.stderr, level='TRACE') # Uncomment to enable verbose logging
logger.add(sys.stderr, level='INFO')

IMG_FILE = Path(os.environ.get('CI_PROJECT_DIR', '.')) / 'data' / 'sun.png'
if not IMG_FILE.exists(): raise FileNotFoundError(f"Image file {IMG_FILE} does not exist")
RESULTS_DIR = Path(os.environ.get('CI_PROJECT_DIR', '.')) / '.cache'
if not RESULTS_DIR.exists(): raise FileNotFoundError(f"Results directory {RESULTS_DIR} does not exist")
SMALLEST_VIEWPORT_EDGE_SIZE = 256

def save_img(img, filename):
  """Save an Image to a File."""
  with open(RESULTS_DIR / (filename + '.png'), 'wb') as f:
    f.write(cv2.imencode('.png', img)[1])

def _json_default(obj):
  if isinstance(obj, tuple):
    return list(obj)
  return obj

def save_json(data, filename):
  """Save a JSON Object to a File."""
  with open(RESULTS_DIR / (filename + '.json'), 'wb') as f:
    f.write(orjson.dumps(data, default=_json_default))

# Load the Original Image
full_img = cv2.imread(str(IMG_FILE), cv2.IMREAD_GRAYSCALE)

# Downsample the Image so the shortest edge is 512px
shortest_edge = min(full_img.shape)
scale_factor = SMALLEST_VIEWPORT_EDGE_SIZE / shortest_edge
img = cv2.resize(full_img, (0, 0), fx=scale_factor, fy=scale_factor)
save_img(img, '0000-sun-downsampled-grayscale')

### Use difference of Guassian to determine the edge of the sun ###

# Apply thresholding to the image to clamp values
_, bin_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
save_img(bin_img, '0010-sun-thresh')
# Apply Gaussian blur to the image
blur_1 = cv2.GaussianBlur(bin_img, (3, 3), 0)
save_img(blur_1, '0020-sun-gaussblur-1')
blur_2 = cv2.GaussianBlur(bin_img, (9, 9), 0)
save_img(blur_2, '0021-sun-gaussblur-2')
# edge = blur_1 - blur_2
edge = blur_2 - blur_1
save_img(edge, '0022-sun-edge-dog')

# Threshold the image to get the edge of the sun: https://docs.opencv.org/3.4/d7/d4d/tutorial_py_thresholding.html
_, edge_mask = cv2.threshold(edge, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # Otsu's Binarization will determine the threshold value automatically
save_img(edge_mask, '0023-sun-edge-mask-dog')

### Find the Center of the Sun ###

"""NOTE

Given the Edge Mask, we want to find the center of the sun & it's radius.

We make a few assumptions:
  
  1. The sun is fully or partially in frame
  2. The majority of pixels in the mask are the sun's edge 
  3. The sun's edge is a continuous line

To find the center of the sun:

  1. Find the contours of the Edge Mask
    - Isolate the largest contour
  2. For a full circle, fit a circle to the contour
  3. For a parrial circle (arc), fit a ...
  4. From the circle, calculate the radius and absolute center
    - clamp the center to the image's dimensions

"""

# Find the contours of the edge mask
edge_mask_points = np.argwhere(edge_mask > 0)[:, [1, 0]]
logger.debug(f"{edge_mask_points.shape=}")
logger.debug(f"{edge_mask_points[0]}")
# raise NotImplementedError
contours, _ = cv2.findContours(edge_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour = max(contours, key=cv2.contourArea)
logger.debug(f"{contour=}")

# Fit a circle to the contour
if len(contours) > 1:
  raise NotImplementedError("Multiple Contours Detected")

# center, radius = cv2.minEnclosingCircle(contour)
(x, y), radius = cv2.minEnclosingCircle(edge_mask_points)
# sun_center_abs = (int(x), int(y))
sun_center_abs = (int(x), int(y))
sun_radius_abs = int(radius)
logger.debug(f"{sun_center_abs=}, {sun_radius_abs=}")
_edge_mask_overlay = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)
cv2.circle(_edge_mask_overlay, sun_center_abs, sun_radius_abs, (0, 255, 0), 1)
save_img(_edge_mask_overlay, '0024-sun-edge-mask-overlay')

# Convert the center to a relative position
sun_center_scale = (
  x / edge_mask.shape[0], # The X-Coordinate (Width)
  y / edge_mask.shape[1], # The Y-Coordinate (Height)
)
logger.debug(f"{sun_center_scale=}")
calc_sun_center = lambda x, y: (int(x * sun_center_scale[0]), int(y * sun_center_scale[1]))
sun_radius_scale = ( # (Assuming 45°) Scalar Projection of the Radius on each Axis
  sun_radius_abs * math.cos(math.pi / 4) / edge_mask.shape[0], # X-Coordinate (Width)
  sun_radius_abs * math.sin(math.pi / 4) / edge_mask.shape[1], # Y-Coordinate (Height)
) # TODO: Can I just Scale this against a single axis?
logger.debug(f"{sun_radius_scale=}")
calc_sun_radius = lambda x, y: round(math.sqrt((x * sun_radius_scale[0])**2 + (y * sun_radius_scale[1])**2))
save_json({
  'sun_center_scale': sun_center_scale,
  'sun_radius_scale': sun_radius_scale,
}, '0030-sun-results')

### For Visualization, Draw the Circle on the Image ###

viz = cv2.imread(str(IMG_FILE), cv2.IMREAD_COLOR)
assert viz.shape[:2] == full_img.shape[:2]
sun_center = calc_sun_center(*full_img.shape[:2])
sun_radius = calc_sun_radius(*full_img.shape[:2])
logger.debug(f"{sun_center=}")
logger.debug(f"{sun_radius=}")
cv2.circle(
  viz,
  sun_center,
  sun_radius,
  (0, 255, 0), # Color: Green
  4, # Line Thickness
)
cv2.circle( # Create a smaller circle at the center to denote the center
  viz,
  sun_center,
  int(2 * sun_radius / 100), # Radius (2% of the Sun's Radius)
  (0, 255, 0), # Color: Green
  -1, # Fill the Circle
)
# Annotate the Center Point w/ it's Coordinates
cv2.putText(
  viz,
  f"({sun_center[0]}, {sun_center[1]})",
  [ # Offset 1% from the point
    sun_center[0] + round(1.5 * sun_center[0] / 100),
    sun_center[1] + round(1.5 * sun_center[1] / 100),
  ],
  cv2.FONT_HERSHEY_SIMPLEX,
  1, # Font Scale
  (0, 255, 0), # Color: Green
  2, # Thickness
  cv2.LINE_AA,
)
save_img(viz, '0030-sun-edge-circle')

# Save the computed results
save_json({
  'sun_center': sun_center,
  'sun_radius': sun_radius,
}, '0040-sun-results')

logger.success("Sun Detection Complete")