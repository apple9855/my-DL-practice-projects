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

| Notebook | Dataset | Notes |
|----------|---------|-------|
| `001` | Bird images from DuckDuckGo | Collected via `fastdownload` + custom labeling |
| `002` | Oxford-IIIT Pet Dataset | Custom binary split (cat vs. non-cat) |
| `003` | PASCAL VOC 2007 / BIWI Head Pose | Multi-label Image Classification + 3D Pose Image Regression |
| `005` | Imagenette (subset of ImageNet) | SOTA techniques - Progressive resizing, Mixup, TTA, Label smoothing |
| `007` | Paddy Disease Classification (Kaggle) | Multi-class fine-grained leaf classification, model comparisons |

---

## 🧠 Model Architectures Used

- **CNNs**: `ResNet18`, `ResNet50`, `EfficientNetV2`, `XResNet`, `RegNetY`
- **Vision Transformers**: `ViT-B/16`, `Swin V2` (Tiny, Small)
- **Custom TIMM backbones**: integrated via `FastAI` `vision_learner`
- **Ensemble Models**: `Swin` + `ViT` for SOTA Kaggle score

---

## 🔧 Skills Demonstrated

- ✅ FastAI `DataBlock` API for dynamic image pipelines  
- ✅ Transfer Learning & Fine-tuning on pretrained models  
- ✅ Multi-label modeling with `BCEWithLogitsLoss`  
- ✅ Image regression & coordinate transformation  
- ✅ Performance tuning: Progressive Resizing, MixUp, Label Smoothing  
- ✅ Inference Boosting: TTA (Test Time Augmentation)  
- ✅ Visual analysis: Confusion Matrix, Prediction plots  
- ✅ Model benchmarking & Kaggle submission flow  

---

## 📚 References

- fastai Book / Course: [`fastbook`](https://github.com/fastai/fastbook)
- TIMM Library: [`rwightman/pytorch-image-models`](https://github.com/rwightman/pytorch-image-models)
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