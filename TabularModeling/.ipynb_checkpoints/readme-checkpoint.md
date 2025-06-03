# 📊 Tabular Modeling with PyTorch, fastai, and Tree-Based Methods - Pratice Showcases

This folder presents four diverse machine learning projects tackling structured (tabular) datasets, each designed to solve a different prediction task with tailored modeling strategies — from binary classification to regression and recommender systems. It demonstrates the use of both low-level PyTorch and high-level fastai APIs, as well as tree-based ML models and neural network architectures.


## 📁 Projects Included

| Notebook | Title | Task Type |
|----------|-------------------------------|-----------------------------|
| `004_titanic_tabular_regression.ipynb` | Titanic Survivor Prediction | Binary Classification |
| `006_1_Bulldozers_saleprices_regression_fastai.ipynb` | Bulldozer Price Estimation | Tabular Regression (Tree + NN) |
| `006_2_houseprice_regression_boosting_ensemble.ipynb` | House Price Prediction (Kaggle) | Tabular Regression (Boosting + Ensemble) |
| `006_3_movie_recom_demo/` | Movie Recommendation Demo | Collaborative Filtering (Recommender System) |



## 🧱 Structure & Dataset Summary

| Notebook | Dataset | Notes |
|----------|---------|-------|
| `004` | Titanic (Kaggle) | Binary label (survived); manual PyTorch training; tabular learner & Random Forest |
| `006_1` | Blue Book for Bulldozers (Kaggle) | Time-aware split; fastai + ensemble trees; RMSE benchmark |
| `006_2` | Ames Housing (Kaggle) | Heavy feature engineering; boosting (XGBoost, LightGBM, CatBoost); top 800+ score |
| `006_3` | MovieLens 100K | User–item interaction matrix; embedding-based models; Gradio demo for deployment |



## 🧠 Model Architectures Used

- **Tree-based Models**: Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost  
- **Neural Network Models**:
  - Custom PyTorch linear model (Titanic)
  - fastai tabular learners with embedding layers  
  - Deep feedforward MLP for regression
  - Embedding-based collaborative filtering (Matrix Factorization, Deep MLP)  
- **Ensembles**:
  - Tree + NN
  - Multiple Boosting models (XGB + CatBoost)



## 🛠️ Skills Demonstrated

- 🧮 From-scratch gradient descent implementation (Titanic)
- 🧠 Neural network modeling for tabular data (with entity embeddings)
- 🌲 Tree-based model tuning & grid search (Bulldozers, Housing)
- 💡 Feature engineering from EDA insights
- 🧪 Time-aware validation splits
- 📉 Evaluation with accuracy, RMSE, MAE
- 🚀 Gradio app deployment (MovieLens recommender)
- 📦 Model ensembling and performance benchmarking



## 🔗 References & Source Materials

- [fastbook](https://github.com/fastai/fastbook) — chapters on tabular and collaborative filtering  
- [Kaggle Datasets & Notebooks](https://www.kaggle.com/) — for Titanic, Bulldozers, Ames Housing  
- [MovieLens](https://grouplens.org/datasets/movielens/) — collaborative filtering dataset  
- fastai v2 library (with `TabularPandas`, `tabular_learner`, `collab_learner`)  
- PyTorch core API for custom model development



## ♻️ Reusability & Adaptability

These projects are modular and serve as templates for:

- Building robust pipelines for real-world tabular data  
- Benchmarking model families (ML vs DL)  
- Educational demonstrations for custom modeling  
- Extending to other domains such as health data, finance, and product recommendation

> 📌 Each notebook is documented, experiment-driven, and designed for readability and modification.