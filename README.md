# Tension_aware_Wire_Tracker

# 1. Preparation
* Clone or download the code package from this repository.
* Set the working directory:
```python
MainFolder = "/your/custom/path"
```

Experimental video of the tensioning strand should be placed in the `Input_Video/` folder.

# 2. Run the Main Pipeline
Open and execute the notebook:
```python
Package.ipynb
```
This notebook will guide you through the full processing pipeline.

# Processing Steps
Step 1: Semantic Segmentation (Sengmentation Anything Model V2)
* Segment the strands using the SAM2 framework.
* Output: Preliminary masks of individual strands.

Step 2: Depth Estimation (Depth Anything V2)
* Use a monocular depth estimator to generate depth maps.
* Compute the ROI by intersecting:
    * SAM2 segmentation mask
    * Valid depth region
* This ensures a more precise ROI for subsequent wire tracking and analysis.

Step 3: Feature Tracking (CoTracker V3)
* Track the displacement of strands across frames using the CoTracker algorithm.
* Output: Time-series of strand displacements.
  
| Step | Module | GitHub | Version / Commit | License |
|------|------|--------|-------------|----------|
| 1 | **SAM 2** – Segment Anything Model V2 | [`facebookresearch/sam2`](https://github.com/facebookresearch/sam2) | sam2.1 | Apache 2.0 |
| 2 | **Depth Anything V2** | [`DepthAnything/Depth-Anything-V2`](https://github.com/DepthAnything/Depth-Anything-V2) | vitl | MIT |
| 3 | **CoTracker** | [`facebookresearch/co-tracker`](https://github.com/facebookresearch/co-tracker) | CoTracker V3 | Apache 2.0 |


## Citation  
The citation information will be available soon.
