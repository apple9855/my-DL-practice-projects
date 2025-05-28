# 🧠 Computer Vision Module – Deep Learning Practice Showcase

This module presents a series of end-to-end computer vision projects covering **image classification**, **multi-label learning**, **image regression**, and **model optimization techniques**, with a focus on **practical implementation** using PyTorch, FastAI, and TIMM.

---

## 📂 Projects Included

| Notebook | Title | Task Type |
|----------|-------|-----------|
| `001_imgcls_bird_classifier.ipynb` | 🐦 Bird-or-Not Classifier | Binary Classification |
| `002_imgcls_oxfordpets_binary.ipynb` | 🐈 Oxford-IIIT Pet – Binary Classification (Cat vs Non-cat) | Binary Classification |
| `003_multilabel_imgreg_timm.ipynb` | 📸 Multi-label Classification + Image Regression with TIMM | Multi-label Classification / Image Regression |
| `005_imgcls_imagenette_optim.ipynb` | Optimizing Image Classification on Imagenette with fastai: From Baseline to SOTA Techniques | Classification + Augmentation |
| `007_imgcls_paddy_disease_benchmark.ipynb` | 🌾 Benchmarking CNNs and Vision Transformers for Paddy Disease Classification (Kaggle) | SOTA Models Benchmarking + Kaggle Submission |

---

## 🧱 Structure & Dataset Summary

| Notebook | Dataset | Task Type | Notes |
|----------|---------|-----------|-------|
| `001` | DuckDuckGo image search (custom) | Binary Classification | Bird classifier with minimal fastai pipeline |
| `002` | Oxford-IIIT Pet Dataset | Binary Classification | Cat vs. Non-Cat split with pretrained ResNet & FastAI |
| `003` | PASCAL VOC 2007 / BIWI Head Pose | Multi-label Classification & Image Regression | TIMM-based CNNs for multi-object detection + 2D/3D head pose prediction |
| `005` | Imagenette (subset of ImageNet) | Image Classification Optimization | Baseline vs. SOTA training techniques (Mixup, TTA, Label Smoothing) |
| `007` | Paddy Disease Classification (Kaggle) | Multi-class Fine-grained Classification | CNN vs. Vision Transformers; Ensemble modeling for leaderboard performance |
---

## 🧠 Model Architectures Used

- **CNNs**: `ResNet18`, `ResNet50`, `EfficientNetV2`, `XResNet`, `RegNetY`
- **Vision Transformers**: `ViT-B/16`, `Swin V2` (Tiny, Small)
- **Custom TIMM backbones**: integrated via `FastAI` `vision_learner`
- **Ensemble Models**: `Swin` + `ViT` for SOTA Kaggle score

---

## 🔧 Skills Demonstrated

- ✅ **Image classification workflows** using `fastai`'s `DataBlock` API  
- ✅ **Transfer learning** and fine-tuning across multiple pretrained backbones (CNNs, ViTs, Swin, RegNetY)  
- ✅ **Multi-label classification** with `BCEWithLogitsLossFlat` and threshold calibration  
- ✅ **Image regression** with pixel-to-real-world coordinate conversion (BIWI head pose)  
- ✅ **Training optimizations**: Progressive Resizing, MixUp, Label Smoothing  
- ✅ **Inference enhancements**: Test Time Augmentation (TTA) and ensemble voting  
- ✅ **Model comparison & benchmarking** across architecture families (ResNet vs ViT vs Swin)  
- ✅ **Compute-aware model selection**: Adaptation for MPS backend (Apple Silicon)  
- ✅ **Kaggle-style development cycle**: From EDA to submission with leaderboard evaluation  
- ✅ **Result visualization & error analysis**: Confusion matrix, prediction overlay plots

---

## 📚 References

- fastai [`Doc`](https://docs.fast.ai/) & Book / Course: [`fastbook`](https://github.com/fastai/fastbook)
- TIMM Library: [`rwightman/pytorch-image-models`](https://github.com/rwightman/pytorch-image-models)
- torchvision [`Table of all available classification weights`](https://docs.pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights)
- Papers:
  - Swin Transformer: [Liu et al. (2021)](https://arxiv.org/abs/2103.14030)
  - ViT (SWAG pretrained): [Wightman et al. (2021)](https://arxiv.org/abs/2110.03599)
  - RegNet: [Radosavovic et al. (2020)](https://arxiv.org/abs/2003.13678)

---

## 🔁 Reusability & Adaptability

Each notebook is designed as a **modular, reproducible pipeline** and can be easily adapted to:

- Custom datasets with minor changes in `DataBlock`
- Alternate architectures via `timm` or `torchvision` backbones
- Vision tasks beyond classification, such as regression, object presence, or localization
- Benchmarking experiments or educational demos

📁 All models are integrated using Hugging Face-compatible and FastAI-friendly workflows.