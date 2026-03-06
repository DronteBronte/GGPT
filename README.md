# GGPT: Geometry-grounded Point Transformer

[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue.svg)](#)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://chenyutongthu.github.io/research/ggpt)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-orange)](https://huggingface.co/YutongGoose/GGPT)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-green)](https://huggingface.co/datasets/YutongGoose/GGPT_eval)



Official PyTorch implementation of **Geometry-grounded Point Transformer (GGPT)** (CVPR 2026). GGPT is a method for high-quality 3D reconstruction from multiview images. For more details, please visit our [project webpage](https://chenyutongthu.github.io/research/ggpt).

---


## 🛠️ Installation

### 1. Clone the repository
```bash
git clone --recursive https://github.com/chenyutongthu/GGPT.git
cd GGPT
```


### 2. Install dependencies

#### 2.0 Create a virtual environment.
#### 2.1 Install torch, torchvision for your CUDA version. (The environment does not require specific CUDA or pytorch version. It has been tested in CUDA 12.1/12.3 and torch 2.2.0/2.5.1.)

#### 2.2 Install requirements for VGGT and SfM.
```
pip install -r requirements_sfm.txt
# Choose which matcher to use.
# For RoMaV2
cd RoMaV2/ && pip install -e .
# For RoMaV1
pip install fused-local-corr>=0.2.2
cd RoMa/ && pip install -e .
```

#### 2.3 \[Optional\] If you need to run GGPT, the 3D point transformer, please follow [the script](ptv3_env.sh) install the following packages in the same virtual environement. You don't need to build another env for this.

#### 2.4 Download our pretrained GGPT checkpoint directly [here](https://huggingface.co/YutongGoose/GGPT).

---

## 📖 Usage & Examples



```bash

python run_demo.py image_dir=/path/to/your/images

```

The outputs (including the feedforward points, SfM points, and the final GGPT points `ggpt_points.ply`) will be saved in the `outputs/demo/` directory by default.


Or you can add ```common_config.ggpt_refine=False``` disable GGPT refinement and run SfM to only obtain the sparse reconstruction.


### ⚙️ Configuration Settings (SfM)

You can adjust Structure-from-Motion (SfM) configuration blocks in the `.yaml` files (found under `configs/`) to better suit your data. 

```yaml
ba_config:
  shared_camera: True # Set it to False if images are captured with different camera intrinsics

dlt_config:
  # Adjust the filtering parameters if you need more accurate yet sparser SfM points. E.g.:
  score_thresh: 0.1       # Increase the matching confidence threshold to filter out noisy matchings
  cycle_err_thresh: 4     # Reduce the cycle error threshold to filter out noisy tracks
  max_epipolar_error: 4   # Reduce the epipolar error threshold to filter out noisy tracks
  min_tri_angle: 3        # Increase the triangulation angle threshold to filter out points with low parallax
  max_reproj_error: 4     # Reduce the reprojection error threshold to filter out noisy points
```

---

## 📊 Evaluation

To run evaluations on the benchmark:

**1. Download the Dataset:**
Download the [preprocessed evaluation set](https://huggingface.co/datasets/YutongGoose/GGPT_eval) and place it at the root of this project as `GGPT_eval/`.

**2. Run Evaluation:**
```bash
sh benchmark_eval.sh
```


## 🧗‍♀️ TODO

* [ ] Release scripts for large-scale multi-view processing.

* [ ] Release GGPT training code.

* [ ] Launch interactive online demo.

## 📖 Citing

If you find our work useful, please cite:

```bibtex
@inproceedings{chen2026ggpt,
  title={GGPT: Geometry-Grounded Point Transformer},
  author={Chen, Yutong and Wang, Yiming and Zhang, Xucong and Prokudin, Sergey and Tang, Siyu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
