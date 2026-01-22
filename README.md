# ShadowDetectionAttached

**Physics-Grounded Attached Shadow Detection Using Approximate 3D Geometry and Light Direction**  
Shilin Hu, Jingyi Xu, Sagnik Das, Dimitris Samaras, Hieu Le  
arXiv, 2025. [[arXiv](https://arxiv.org/abs/2512.06179)]

This repository currently provides **qualitative results**.  
**Code and training/inference instructions will be released in a future update.**

---

## Overview

We introduce a framework that jointly detects cast and attached shadows by reasoning about their mutual relationship with scene illumination and geometry.

- **Shadow detection module:** predicts both shadow types separately.  
- **Light estimation module:** infers the light direction from the detected shadows and inputs.

The estimated light direction, combined with **surface normals**, produces a geometry-consistent map of likely self-occluded regions, which is fed back to iteratively refine both shadow segmentation and light estimation.

![Overview](assets/overview.png)

---

## Results

### Image Results

Representative qualitative results on single images:

<figure>
  <img src="assets/qual.png" alt="Qualitative Results" width="80%"/>
  <figcaption><em>Qualitative results on our dataset. From left to right: input image, BDRAR, FSDNet, FDRNet, SILT, Ours and GT prediction (Green: Attached, Red: Cast).</em> </figcaption>
</figure>

### TimeLapse Results

Qualitative time-lapse results:

<p align="center">
  <img src="assets/brick.gif" width="60%" /><br/><br/>
  <img src="assets/pottedplant.gif" width="60%" /><br/><br/>
  <img src="assets/sunclock.gif" width="60%" />
</p>
<p align="center">
  <em>Time-lapse qualitative results (top to bottom): Brick, Potted Plant, Sun Clock.</em>
</p>

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{hu2025physicsgroundedattachedshadowdetection,
  title         = {Physics-Grounded Attached Shadow Detection Using Approximate 3D Geometry and Light Direction},
  author        = {Shilin Hu and Jingyi Xu and Sagnik Das and Dimitris Samaras and Hieu Le},
  year          = {2025},
  eprint        = {2512.06179},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2512.06179}
}
