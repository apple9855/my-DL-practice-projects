# 🎬 Movie Recommendation System (MovieLens 100K)

A collaborative filtering-based movie recommender built with **fastai** and **PyTorch**, trained on the classic [MovieLens 100K dataset](https://grouplens.org/datasets/movielens/100k/), and deployed with **Gradio** on **Hugging Face Spaces**.



## 🚀 Highlights

- **Collaborative Filtering**: Implemented dot-product and neural-based recommenders  
- **Deep Dive**: Manual use of `nn.Parameter` and `nn.Embedding` to explore latent factors  
- **Fastai Integration**: Training via `CollabLearner` with flexible model experimentation  
- **Deployment**: Live Gradio app for top-N movie recommendation  




## 🖥️ Demo

🔗 [Try it live →](https://huggingface.co/spaces/apple9855/movie-recommend)  
Input a user ID and get 5 personalized movie recommendations!



## 📂 Files for Deployment

- `app.py` — Gradio interface  
- `export.pkl` — Trained model  
- `ratings.csv` — Processed dataset  
- `requirements.txt` — Environment config (`fastai==2.7.12` required)



## 👩‍💻 Author Note

This project was built as part of a deep learning engineering portfolio, highlighting:

- Hands-on embedding model construction  
- Neural collaborative filtering experimentation  
- Production-ready ML app deployment