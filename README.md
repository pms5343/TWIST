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
Run.ipynb
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

Lay Angle Estimation
* The lay angle is estimated using LDC edge detection applied only to the first frame of the video.

  
| Step | Module | GitHub | Version / Commit | License |
|------|------|--------|-------------|----------|
| 1 | **SAM 2** – Segment Anything Model V2 | [`facebookresearch/sam2`](https://github.com/facebookresearch/sam2) | sam2.1 | Apache 2.0 |
| 2 | **Depth Anything V2** | [`DepthAnything/Depth-Anything-V2`](https://github.com/DepthAnything/Depth-Anything-V2) | vitl | MIT |
| 3 | **CoTracker** | [`facebookresearch/co-tracker`](https://github.com/facebookresearch/co-tracker) | CoTracker V3 | Apache 2.0 |

# Contributors
<p>
  <strong>Dongyoung Ko</strong>
  <a href="https://scholar.google.com/citations?user=uJ5Ot9kAAAAJ&hl=en">
    <img src="https://img.shields.io/badge/-4285F4?style=flat&logo=googlescholar&logoColor=white" alt="Google Scholar"/>
  </a>
  <a href="https://github.com/ehddud3555-skku">
    <img src="https://img.shields.io/badge/-000000?style=flat&logo=github&logoColor=white" alt="GitHub"/>
  </a>
</p>

<p>
  <strong>Minsoo Park</strong>
  <a href="https://scholar.google.com/citations?user=6dCUM5oAAAAJ&hl=En">
    <img src="https://img.shields.io/badge/-4285F4?style=flat&logo=googlescholar&logoColor=white" alt="Google Scholar"/>
  </a>
  <a href="https://github.com/pms5343">
    <img src="https://img.shields.io/badge/-000000?style=flat&logo=github&logoColor=white" alt="GitHub"/>
  </a>
</p>

<p>
  <strong>Taebum Lee</strong>
  <a href="https://github.com/pms5343">
    <img src="https://img.shields.io/badge/-000000?style=flat&logo=github&logoColor=white" alt="GitHub"/>
  </a>
</p>


<p>
  <strong>Soojin Jin</strong>
  <a href="https://scholar.google.com/citations?user=6dCUM5oAAAAJ&hl=En">
    <img src="https://img.shields.io/badge/-4285F4?style=flat&logo=googlescholar&logoColor=white" alt="Google Scholar"/>
  </a>
  <a href="[https://github.com/pms5343](https://github.com/sujin1229)">
    <img src="https://img.shields.io/badge/-000000?style=flat&logo=github&logoColor=white" alt="GitHub"/>
  </a>
</p>



## Citation  
The citation information will be available soon.
