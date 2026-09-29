# AttachedShadowDetection

**Cast and Attached Shadow Detection via Iterative Light and Geometry Reasoning**  
Shilin Hu, Jingyi Xu, Sagnik Das, Dimitris Samaras, Hieu Le
  
**[[Paper](https://arxiv.org/abs/2512.06179)] [[Project Page](https://shilin21.github.io/attached_detection/)]**

**Accepted to ECCV 2026.**

**Updates:**

- **09-29-2026:** Released the training and testing code.
- **09-10-2026:** Added a corrected evaluation protocol for future comparisons while retaining the reported protocol for paper reproduction.
- **09-09-2026:** Updated the dataset with additional mask corrections.

---

## Overview

We introduce a framework that jointly detects cast and attached shadows by reasoning about their mutual relationship with scene illumination and geometry.

- **Shadow detection module:** predicts both shadow types separately.
- **Light estimation module:** infers the light direction from the detected shadows and inputs.

The estimated light direction, combined with **surface normals**, produces a geometry-consistent map of likely self-occluded regions, which is fed back to iteratively refine both shadow segmentation and light estimation.

![Overview](assets/overview.png)

---

## Dataset

[Download the Cast and Attached Shadow Dataset](https://drive.google.com/drive/folders/1fVUd2Srj9Kaf-3ytJLmCf7kGRQY3Pvic?usp=sharing)

This dataset contains 1,458 images derived from WSRD, SOBA, and CUHK-Shadow and is provided for noncommercial academic research. The original source datasets retain their respective rights and terms: [WSRD](https://github.com/fvasluianu97/WSRD-DNSR) is under CC BY-NC-SA 4.0, [CUHK-Shadow](https://github.com/xw-hu/CUHK-Shadow) is under CC BY-NC-SA 3.0, and [SOBA](https://github.com/stevewongv/SSIS) is distributed through its official public research release. Please cite accordingly when using this dataset.

---

## Usage

Python 3.9 is recommended. The code was tested with Python 3.9.17, PyTorch 1.13.1, torchvision 0.14.1, and CUDA 11.7.

```bash
conda create -n attached-shadow python=3.9
conda activate attached-shadow
pip install -r requirements.txt
```

## Training

Training uses the Cast–Attached Shadow Dataset and the SBU-trained PVT-B5 initialization from [SILT](https://github.com/hanyangclarence/SILT).

Set `data_root`, `backbone_checkpoint`, and `output_dir` in `configs/train.yaml`, then run:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train.sh
```

The paths can also be specified from the command line:

```bash
bash scripts/train.sh \
  --data_root /path/to/ECCV_castattached \
  --backbone_ckpt /path/to/SILT_PVTb5_SBU.pth \
  --output_dir /path/to/output
```

## Evaluation

[Download the final model checkpoint](https://drive.google.com/drive/folders/1fVUd2Srj9Kaf-3ytJLmCf7kGRQY3Pvic?usp=sharing)

Set `data_root`, `model_checkpoint`, and `output_dir` in `configs/test.yaml`, then run:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/test.sh
```

The paths can also be specified from the command line:

```bash
bash scripts/test.sh \
  --data_root /path/to/ECCV_castattached \
  --ckpt /path/to/attached_shadow_detection_eccv26.pth \
  --save_dir /path/to/results
```

The evaluator provides two protocols:

- `standard` is now the **default** and is **recommended** for comparisons with other methods. It binarizes ground-truth masks at 0.5 and computes metrics from conventional binary confusion matrices.
- `reported` reproduces the values reported for our method in the paper. It retains fractional boundary values rather than binarizing them, so boundary pixels contribute differently to the metrics. Its results therefore differ from the standard protocol.

Select the protocol with:

```bash
bash scripts/test.sh --protocol standard
bash scripts/test.sh --protocol reported
```

Results using the corrected standard evaluation protocol:

Best results are **bold** and second-best results are *italicized*.

| Method | Full BER↓ | Full F1↑ | Cast BER↓ | Cast F1↑ | Attached BER↓ | Attached F1↑ |
|---|---:|---:|---:|---:|---:|---:|
| BDRAR | 21.31 | 70.51 | 7.11 | 79.82 | 35.31 | 52.37 |
| DSD | 22.55 | 68.73 | 7.70 | 80.27 | 35.83 | 51.78 |
| FSDNet | 24.47 | 66.08 | 10.38 | 79.35 | 37.42 | 46.29 |
| MTMT | 33.11 | 49.96 | 15.04 | 74.84 | 41.93 | 38.12 |
| FDRNet | 23.71 | 65.67 | 8.83 | 71.81 | 34.72 | 55.19 |
| SDCM | 24.29 | 66.48 | 7.75 | **83.58** | 36.73 | 49.06 |
| SILT | 20.22 | 71.49 | *4.51* | 80.56 | 32.60 | 60.29 |
| BDRAR† | 8.14 | 83.98 | 5.05 | 73.07 | 20.65 | *82.79* |
| FSDNet† | 10.17 | **85.12** | 5.69 | 82.16 | *19.39* | 80.66 |
| FDRNet† | 8.62 | 83.03 | *4.51* | 73.48 | 23.02 | 81.52 |
| SILT† | **6.49** | 83.07 | **4.37** | 68.62 | 26.42 | 80.85 |
| Ours | *7.95* | *84.46* | 6.33 | *82.79* | **16.60** | **83.76** |

† Fine-tuned on the training split.

By default, evaluation prints all full-shadow, cast-shadow, attached-shadow, and light-direction metrics and saves visual prediction panels. To report metrics without saving images, run:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/test.sh --metrics-only
```

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
