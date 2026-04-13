## OVERVIEW
Python computer vision project for liquid fill defect detection using OpenCV.

## STRUCTURE
```
cv_handel/
├── sample/                # Test images by plate type
│   ├── 96-50-黑色/        # 96-well, 50µL, black liquid
└── output/                # Generated detection results (gitignore)
```

## CONVENTIONS
- **Naming**: `snake_case` functions/vars, `PascalCase` classes, `UPPER_SNAKE_CASE` constants
- **Imports**: stdlib → third-party (cv2, numpy)
- **Docstrings**: Chinese for user-facing descriptions
- **Linter**: Ruff (default config, `.ruff_cache/` present)
- **Main guard**: All scripts use `if __name__ == "__main__":`


## UNIQUE STYLES
- Sample naming: `{plate_type}-{volume}-{color}` (e.g., `96-50-黑色`)
- Image naming: `{plate}-{number}.jpg` (1=empty, 2=test, 参照=reference)
- Detection: Grid-based ROI brightness threshold (not contour-based)
- Morphological ops: ellipse kernel, dilate(2) + erode(1)

## COMMANDS
```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install opencv-python numpy

```

## NOTES
- **Missing**: requirements.txt, README.md, .gitignore, pyproject.toml
- **Hardcoded paths**: Scripts use `/Users/hedengfeng/company/cv_handel/...`
- **Plate types**: 96-well (50µL, 1000µL), 384-well (50µL, 125µL)
- **Python**: 3.13.3 in venv/