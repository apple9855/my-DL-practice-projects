# 🧠 Computer Vision Module – Deep Learning Practice Showcase

This module presents a series of end-to-end computer vision projects covering **image classification**, **multi-label learning**, **image regression**, and **model optimization techniques**, with a focus on **practical implementation** using PyTorch, FastAI, and TIMM.


## 📂 Projects Included

| Notebook | Title | Task Type |
|----------|-------|-----------|
| `001_imgcls_bird_classifier.ipynb` | 🐦 Bird-or-Not Classifier | Binary Classification |
| `002_imgcls_oxfordpets_binary.ipynb` | 🐈 Oxford-IIIT Pet – Binary Classification (Cat vs Non-cat) | Binary Classification |
| `003_multilabel_imgreg_timm.ipynb` | 📸 Multi-label Classification + Image Regression with TIMM | Multi-label Classification / Image Regression |
| `005_imgcls_imagenette_optim.ipynb` | Optimizing Image Classification on Imagenette with fastai: From Baseline to SOTA Techniques | Classification + Augmentation |
| `007_imgcls_paddy_disease_benchmark.ipynb` | 🌾 Benchmarking CNNs and Vision Transformers for Paddy Disease Classification (Kaggle) | SOTA Models Benchmarking + Kaggle Submission |


## 🧱 Structure & Dataset Summary

| Notebook | Dataset | Task Type | Notes |
|----------|---------|-----------|-------|
| `001` | DuckDuckGo image search (custom) | Binary Classification | Bird classifier with minimal fastai pipeline |
| `002` | Oxford-IIIT Pet Dataset | Binary Classification | Cat vs. Non-Cat split with pretrained ResNet & FastAI |
| `003` | PASCAL VOC 2007 / BIWI Head Pose | Multi-label Classification & Image Regression | TIMM-based CNNs for multi-object detection + 2D/3D head pose prediction |
| `005` | Imagenette (subset of ImageNet) | Image Classification Optimization | Baseline vs. SOTA training techniques (Mixup, TTA, Label Smoothing) |
| `007` | Paddy Disease Classification (Kaggle) | Multi-class Fine-grained Classification | CNN vs. Vision Transformers; Ensemble modeling for leaderboard performance |


## 🧠 Model Architectures Used

- **Classical CNNs**: `ResNet18`, `ResNet50`
- **Efficient CNNs**: `EfficientNetV2`
- **Vision Transformers**: `ViT-B/16`, `Swin V2` (Tiny, Small)
- **Hybrid Conv-Attention**: `XResNet`, `RegNetY`
- **Ensemble**: `Swin` + `ViT` for SOTA Kaggle score
- **Custom TIMM backbones**: integrated via `FastAI` `vision_learner`
>All models are pretrained on ImageNet or SWAG and fine-tuned on task-specific datasets.



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



## 📚 References

- fastai [`Doc`](https://docs.fast.ai/) & Book / Course: [`fastbook`](https://github.com/fastai/fastbook)
- TIMM Library: [`rwightman/pytorch-image-models`](https://github.com/rwightman/pytorch-image-models)
- torchvision [`Table of all available classification weights`](https://docs.pytorch.org/vision/stable/models.html#table-of-all-available-classification-weights)
- Papers:
  - Swin Transformer: [Liu et al. (2021)](https://arxiv.org/abs/2103.14030)
  - ViT (SWAG pretrained): [Wightman et al. (2021)](https://arxiv.org/abs/2110.03599)
  - RegNet: [Radosavovic et al. (2020)](https://arxiv.org/abs/2003.13678)



## 🔁 Reusability & Adaptability

This repository is structured to support modular reuse and adaptation across various computer vision tasks:
- **Data Pipeline**: Built with FastAI’s DataBlock API — customizable for classification, multi-label, and regression tasks.
- **Model-Agnostic Training**: Easily switch between backbones (ResNet, EfficientNet, ViT, Swin, RegNet) using torchvision or timm.
- **Loss & Metric Decoupling**: Swap out loss functions and metrics as needed (CrossEntropyLoss, BCEWithLogitsLossFlat, MSELoss, etc.).
- **Training Tricks**: Plug-and-play support for Progressive Resizing, MixUp, Label Smoothing, and Test Time Augmentation (TTA).
- **Resource Adaptability**: Compatible with CPU, Apple MPS, and GPU backends — optimized for lightweight environments like Mac or Colab.
- **Highly Extendable**: Suitable as a clean starting point for medical imaging, agricultural diagnosis, industrial vision, and more.

>To reuse: simply adapt the dataset, adjust the DataBlock, and fine-tune — no need to rewrite the training pipeline.