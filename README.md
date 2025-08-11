# TWIST: Tension from Wire-image Strand Tracking
## Vision-based Automated Cable Tension Monitoring Using Pixel Tracking
Dongyoung Ko, Minsoo Park∗, Soojin Jin, PA PA WIN AUNG, Seunghee Park∗

## Abstract
Traditional methods for monitoring cable tension rely on indirect measurements such as cable vibrations and often require specialized calibration. These approaches limit the efficiency, and non-contact capability of tension monitoring across various structures. This paper presents a vision-based framework for automated cable tension monitoring, which directly captures image data of internal steel strands. By leveraging advanced computer vision techniques such as zero-shot segmentation, depth estimation, edge detection, and dense pixel tracking critical geometric parameters are extracted and integrated into a kinematic-based model for tension estimation. A calibration-free method for estimating real-world pixel size, derived from the helical geometry of the strands, enables field deployment without the need for camera setup information. Experimental results show strong correlation with reference data, achieving a mean absolute error of 4.94% under elastic conditions. These findings pave the way for a promising alternative in vision-based structural health monitoring for prestressed structures.

## System Requirements
All experiments were conducted under the following hardware and software configuration:

### Hardware Environment
* Workstation: Dell Precision 7920 Rack
* CPU: Intel Xeon Silver 4210R (10 cores, 20 threads, 2.4–3.2 GHz)
* Memory: 128 GB DDR4-3200 ECC
* GPU: NVIDIA RTX A6000 (48 GB GDDR6 ECC)
* Storage: 1 TB NVMe SSD
* Operating System: Ubuntu 20.04 LTS
### Software Environment
* Python: 3.10
* PyTorch: 2.1.0 (with CUDA 12.1)
* TensorFlow: 2.15 (with CUDA 11.7)
* scikit-learn: 1.4
  
# 1. Preparation

* Clone or download the code package from this repository.
* Set the working directory:
```python
MainFolder = "/your/custom/path"
```
Experimental video of the tensioning strand should be placed in the `Input_Video/` folder.

### Depth-Anything

* Download the `depth_anything_v2_vitl.pth` checkpoint file and place it in the `Depth-Anything-V2/checkpoints/` directory.
* The pre-trained checkpoint can be downloaded from:  
  https://github.com/DepthAnything/Depth-Anything-V2#pre-trained-models

### SAM (Segment Anything Model)

* Download the `sam2.1_hiera_large.pt` checkpoint file and place it in the `sam2_repo/checkpoints/` directory.
* The pre-trained checkpoint can be downloaded from:  
  https://github.com/facebookresearch/sam2

### Cotracker

* Download the `scaled_offline.pth` checkpoint file and place it in the `co-tracker/cotracker/checkpoints/` directory.
* The pre-trained checkpoint can be downloaded from:  
  https://github.com/facebookresearch/co-tracker

| | Module | GitHub | Version / Commit | License |
|------|------|--------|-------------|----------|
| 1 | **SAM 2** – Segment Anything Model V2 | [`facebookresearch/sam2`](https://github.com/facebookresearch/sam2) | sam2.1 | Apache 2.0 |
| 2 | **Depth Anything V2** | [`DepthAnything/Depth-Anything-V2`](https://github.com/DepthAnything/Depth-Anything-V2) | vitl | MIT |
| 3 | **CoTracker** | [`facebookresearch/co-tracker`](https://github.com/facebookresearch/co-tracker) | CoTracker V3 | Apache 2.0 |

### LDC (Lightweight Dense CNN for Edge Detection)

* Download code from https://github.com/xavysp/LDC
* For custom usage:
- **[main.py]** Set input image directory:  
  `--input_val_dir=/your/image/folder`

- **[main.py]** Set output directory for results:  
  `--output_dir=/your/output/folder`

- **[main.py]** Match image resolution to your data:  
  `--img_width=YOUR_WIDTH` and `--img_height=YOUR_HEIGHT`  
  (e.g., 1080×710)

- **[img_processing.py]** Adjust edge map threshold:  
  ```python
  tensor = np.where(tensor >= 0.70, tensor, 0)  # threshold for edge map

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
* Run `Edge(LDC).ipynb` to estimate the lay angle for all frames.
 
# Citation  
The citation information will be available soon.

## Contributors
<p>
  <strong>Dongyoung Ko</strong>
  <a href="https://sites.google.com/view/skkuscit" target="_blank">
    <img src="https://github.com/pms5343/Tension_aware_Wire_Tracker/raw/main/logo/skku.svg" height="20" alt="SKKU Logo"/>
  </a>
  <a href="https://scholar.google.com/citations?user=uJ5Ot9kAAAAJ&hl=en" target="_blank">
    <img src="https://img.shields.io/badge/-4285F4?style=flat&logo=googlescholar&logoColor=white" alt="Google Scholar"/>
  </a>
  <a href="https://github.com/ehddud3555-skku" target="_blank">
    <img src="https://img.shields.io/badge/-000000?style=flat&logo=github&logoColor=white" alt="GitHub"/>
  </a>
</p>

<p>
  <strong>Minsoo Park</strong>
  <a href="https://sites.google.com/view/iisc-lab" target="_blank">
    <img src="https://github.com/pms5343/Tension_aware_Wire_Tracker/raw/main/logo/GWNU.svg" height="20" alt="GWNU Logo"/>
  </a>
  <a href="https://scholar.google.com/citations?user=6dCUM5oAAAAJ&hl=En">
    <img src="https://img.shields.io/badge/-4285F4?style=flat&logo=googlescholar&logoColor=white" alt="Google Scholar"/>
  </a>
  <a href="https://github.com/pms5343">
    <img src="https://img.shields.io/badge/-000000?style=flat&logo=github&logoColor=white" alt="GitHub"/>
  </a>
</p>

<p>
  <strong>Taebum Lee</strong>
  <a href="https://smartinside.ai/" target="_blank">
  <img src="https://github.com/pms5343/Tension_aware_Wire_Tracker/raw/main/logo/SI.png" height="20" alt="SmartInside Logo"/>
  <a href="https://github.com/ltb1021">
    <img src="https://img.shields.io/badge/-000000?style=flat&logo=github&logoColor=white" alt="GitHub"/>
  </a>
</p>


<p>
  <strong>Sujin Jin</strong>
  <a href="https://sites.google.com/view/skkuscit" target="_blank">
  <img src="https://github.com/pms5343/Tension_aware_Wire_Tracker/raw/main/logo/skku.svg" height="20" alt="SKKU Logo"/>
  <a href="https://scholar.google.com/citations?view_op=list_works&hl=en&user=SH8aOoEAAAAJ">
    <img src="https://img.shields.io/badge/-4285F4?style=flat&logo=googlescholar&logoColor=white" alt="Google Scholar"/>
  </a>
  <a href="https://github.com/sujin1229">
    <img src="https://img.shields.io/badge/-000000?style=flat&logo=github&logoColor=white" alt="GitHub"/>
  </a>
</p>
    
<p>
  <strong>Seunghee Park</strong>
  <a href="https://sites.google.com/view/skkuscit" target="_blank">
    <img src="https://github.com/pms5343/Tension_aware_Wire_Tracker/raw/main/logo/skku.svg" height="20" alt="SKKU Logo"/>
  </a>
  <a href="https://scholar.google.com/citations?user=_CUQYq8AAAAJ&hl=en" target="_blank">
    <img src="https://img.shields.io/badge/-4285F4?style=flat&logo=googlescholar&logoColor=white" alt="Google Scholar"/>
  </a>
</p>
