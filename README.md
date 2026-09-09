# AttachedShadowDetection

**Cast and Attached Shadow Detection via Iterative Light and Geometry Reasoning**  
Shilin Hu, Jingyi Xu, Sagnik Das, Dimitris Samaras, Hieu Le  
arXiv, 2026. [[arXiv](https://arxiv.org/abs/2512.06179)]

**Accepted to ECCV 2026.**

This repository currently provides the **dataset and qualitative results**.  
**Code and training/inference instructions will be released in a future update.**

---

## Overview

We introduce a framework that jointly detects cast and attached shadows shadows by reasoning about their mutual relationship with scene illumination and geometry.

- **Shadow detection module:** predicts both shadow types separately.
- **Light estimation module:** infers the light direction from the detected shadows and inputs.

The estimated light direction, combined with **surface normals**, produces a geometry-consistent map of likely self-occluded regions, which is fed back to iteratively refine both shadow segmentation and light estimation.

![Overview](assets/overview.png)

---

## Dataset

[Download the Cast and Attached Shadow Dataset](https://drive.google.com/drive/folders/1fVUd2Srj9Kaf-3ytJLmCf7kGRQY3Pvic?usp=sharing)

This dataset contains 1,458 images derived from WSRD, SOBA, and CUHK-Shadow and is provided for noncommercial academic research. The original source datasets retain their respective rights and terms: [WSRD](https://github.com/fvasluianu97/WSRD-DNSR) is under CC BY-NC-SA 4.0, [CUHK-Shadow](https://github.com/xw-hu/CUHK-Shadow) is under CC BY-NC-SA 3.0, and [SOBA](https://github.com/stevewongv/SSIS) is distributed through its official public research release. Please cite accordingly when using this dataset.

---

## Results

### Image Results

Representative qualitative results on single images:

<figure>
  <img src="assets/qual.png" alt="Qualitative Results" width="80%"/>
  <figcaption><em>Qualitative results on our dataset. From left to right: input image, BDRAR, FSDNet, FDRNet, SILT, Ours, and GT (Green: Attached, Red: Cast).</em></figcaption>
</figure>

### Time-Lapse Results

Qualitative time-lapse results:

<p align="center">
  <img src="assets/brick.gif" width="60%"/><br/><br/>
  <img src="assets/pottedplant.gif" width="60%"/><br/><br/>
  <img src="assets/sunclock.gif" width="60%"/>
</p>

<p align="center">
  <em>Time-lapse qualitative results (top to bottom): Brick, Potted Plant, Sun Clock.</em>
</p>

---

## Citation

If you find this work useful, please cite:

```bibtex
@misc{hu2026castattachedshadowdetection,
      title={Cast and Attached Shadow Detection via Iterative Light and Geometry Reasoning},
      author={Shilin Hu and Jingyi Xu and Sagnik Das and Dimitris Samaras and Hieu Le},
      year={2026},
      eprint={2512.06179},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.06179},
}
```
