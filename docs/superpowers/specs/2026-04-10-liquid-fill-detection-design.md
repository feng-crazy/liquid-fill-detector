# Liquid Fill Detection System Design

**Date**: 2026-04-10  
**Plate Type**: 96-well (50µL, black liquid)  
**Approach**: Hybrid Detection + Grid Mapping

---

## Overview

Design a machine vision algorithm to detect liquid fill defects in 96-well microplates with black liquid. The system extracts per-tube analysis data and generates intermediate visualization outputs.

---

## Architecture

### Module Structure

```
cv_handel/
├── detect_liquid_fill.py    # Main entry point
├── lib/
│   ├── grid_detector.py      # Grid structure detection
│   ├── droplet_detector.py   # Black droplet detection
│   ├── tube_mapper.py        # Map droplets to grid positions
│   └── output_writer.py      # JSON + annotated image output
└── output/
    └── {plate_type}-{timestamp}/
        ├── result.json       # Per-tube analysis
        ├── annotated.jpg     # Visual result
        └── stats.json        # Summary statistics
        └── 01-08_*.jpg       # Intermediate images
```

### Key Components

| Component | Purpose | Input | Output |
|-----------|---------|-------|--------|
| GridDetector | Find 8×12 grid boundaries | ROI image | Row/col boundary coordinates |
| DropletDetector | Detect black liquid droplets | Grayscale ROI | Contour list with centroids |
| TubeMapper | Assign grid positions | Contours + grid | Tube status matrix |
| OutputWriter | Generate outputs | Tube matrix | JSON + images |

---

## Data Flow

### Processing Pipeline

```
Input Image (96-50-*.jpg)
    ↓
[1] ROI Extraction → 01_roi.jpg
    ↓
[2] Grayscale + Gaussian Blur → 02_preprocessed.jpg
    ↓
[3] Threshold Segmentation → 03_threshold.jpg
    ↓
[4] Morphological Operations → 04_morphology.jpg
    ↓
[5] Contour Detection → 05_contours.jpg
    ↓
[6] Grid Detection → 06_grid.jpg
    ↓
[7] Tube Mapping → 07_mapping.jpg
    ↓
[8] Final Result → 08_annotated.jpg + result.json + stats.json
```

### Intermediate Images

| Stage | Filename | Description |
|-------|----------|-------------|
| ROI extraction | `01_roi.jpg` | Cropped region excluding top orange strip/screws |
| Preprocessing | `02_preprocessed.jpg` | Grayscale with Gaussian blur (5×5 kernel) |
| Threshold | `03_threshold.jpg` | Binary mask: black regions become white (foreground) |
| Morphology | `04_morphology.jpg` | Cleaned mask: ellipse kernel, dilate(2) + erode(1) |
| Contours | `05_contours.jpg` | All valid contours (area 20-2000 px²) outlined |
| Grid lines | `06_grid.jpg` | Detected row/col boundaries overlaid on ROI |
| Tube mapping | `07_mapping.jpg` | Each contour labeled with row/col coordinate |
| Final | `08_annotated.jpg` | Filled tubes (green circles), empty tubes (red circles), stats text |

---

## Algorithm Details

### ROI Extraction

Crop image to focus on tube tip region:
- Top: 15% removed (orange strip + screws)
- Bottom: 5% removed
- Left: 12% removed
- Right: 12% removed

### Threshold Segmentation

- Convert ROI to grayscale
- Apply Gaussian blur (σ=1, kernel 5×5)
- Threshold: `cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)`
  - Pixels < 60 (dark/black) → white (foreground)
  - Pixels ≥ 60 → black (background)

### Morphological Operations

Per AGENTS.md convention:
- Kernel: ellipse (matches liquid droplet shape)
- Sequence: dilate(iterations=2) → erode(iterations=1)
- Purpose: Fill holes in droplets, remove small noise

### Grid Detection

Projection profile analysis:
1. Horizontal projection: Sum each row → find 8 peaks (row boundaries)
2. Vertical projection: Sum each column → find 12 peaks (col boundaries)
3. Assign row letters: A-H (top to bottom)
4. Assign col numbers: 1-12 (left to right)

Fallback: If projection fails (flat profile), use evenly spaced grid.

### Contour Filtering

```python
min_area = 20   # Filter noise
max_area = 2000 # Filter large artifacts
valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
```

### Tube Mapping

1. Calculate centroid for each valid contour
2. Compare centroid coordinates to grid boundaries
3. Assign row (A-H) and column (1-12)
4. Mark grid positions with no mapped droplet as "empty"

---

## Output Format

### result.json

```json
{
  "plate_info": {
    "type": "96-50-黑色",
    "rows": 8,
    "columns": 12,
    "total_tubes": 96
  },
  "tubes": [
    {
      "row": "A",
      "column": 1,
      "status": "filled",
      "droplet_area": 150,
      "centroid": {"x": 120, "y": 85}
    },
    {
      "row": "C",
      "column": 5,
      "status": "empty",
      "droplet_area": null,
      "centroid": null
    }
  ],
  "timestamp": "2026-04-10T15:30:00"
}
```

### stats.json

```json
{
  "total_tubes": 96,
  "filled_count": 88,
  "empty_count": 8,
  "fill_rate": 91.67,
  "verdict": "FAIL",
  "defect_allowance": 0,
  "processing_time_ms": 245
}
```

### Directory Structure

```
output/96-50-黑色-2026-04-10-153000/
├── result.json
├── stats.json
├── 01_roi.jpg
├── 02_preprocessed.jpg
├── 03_threshold.jpg
├── 04_morphology.jpg
├── 05_contours.jpg
├── 06_grid.jpg
├── 07_mapping.jpg
└── 08_annotated.jpg
```

---

## Error Handling

### Cases Handled

| Case | Detection | Response |
|------|-----------|----------|
| Image load failure | `cv2.imread()` returns None | Exit with error message |
| ROI empty | Crop results in zero area | Use full image, log warning |
| Grid detection failure | Flat projection profile | Use evenly spaced fallback grid |
| No droplets found | Contour count = 0 | Report all 96 empty, FAIL verdict |
| Too many contours | Count > 150 | Apply stricter filtering, log warning |
| Boundary ambiguity | Centroid near grid edge | Assign to nearest cell |

### Configurable Parameters

```python
CONFIG = {
    'threshold_value': 60,      # Black detection sensitivity
    'min_area': 20,             # Minimum droplet size
    'max_area': 2000,           # Maximum droplet size
    'standard_count': 96,       # Expected tube count
    'defect_allowance': 0,      # Allowed empty tubes (0 = zero tolerance)
    'roi_top': 0.15,            # Top crop ratio
    'roi_bottom': 0.95,         # Bottom crop ratio
    'roi_left': 0.12,           # Left crop ratio
    'roi_right': 0.88,          # Right crop ratio
    'gaussian_kernel': 5,       # Blur kernel size
}
```

---

## Testing

### Test Images

| Image | Expected Filled | Expected Empty | Expected Verdict |
|-------|-----------------|----------------|------------------|
| `96-50-1.jpg` (empty) | 0-5 | 91-96 | FAIL |
| `96-50-2.jpg` (full) | ≥90 | 0-6 | PASS |
| `96-50-参照.jpg` (8 defects) | ~88 | 8 | FAIL |

### Verification Checklist

- [ ] All 8 intermediate images generated
- [ ] JSON output contains 96 tube entries (A1-H12)
- [ ] Empty tubes in reference image match red circle positions
- [ ] Droplet count within ±5 of manual count
- [ ] PASS/FAIL verdict matches defect_allowance logic

### Success Criteria

1. Reference image (`96-50-参照.jpg`) detects 8 empty tubes at marked positions
2. Full plate (`96-50-2.jpg`) detects ≥90 droplets
3. Empty plate (`96-50-1.jpg`) detects ≤5 false positives
4. Output JSON valid and parseable
5. Annotated image clearly shows filled (green) vs empty (red) tubes

---

## Dependencies

- **Python**: 3.13.3
- **OpenCV**: `opencv-python` (cv2)
- **NumPy**: `numpy`
- **Standard library**: `json`, `os`, `datetime`

---

## Implementation Notes

- Follow AGENTS.md conventions: snake_case functions, PascalCase classes
- Use `if __name__ == "__main__":` guard for main script
- Docstrings in Chinese for user-facing descriptions
- Imports order: stdlib → third-party (cv2, numpy)
- Ruff linter compliance required