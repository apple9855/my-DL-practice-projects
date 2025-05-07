# 🎬 Movie Recommendation System (MovieLens 100K)

This project builds a movie recommendation system based on the MovieLens 100K dataset using collaborative filtering techniques implemented via the `fastai` library. It demonstrates model development, evaluation, and deployment through a web-based Gradio demo hosted on Hugging Face Spaces.

---

## 🧠 Models & Techniques

- **Dataset**: [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)  
  (943 users, 1682 movies, 100K ratings)

- **Collaborative Filtering Architectures**:
  - `DotProduct`: vanilla dot-product-based embedding model
  - `DotProduct + y_range`: adds output clipping for score stability
  - `DotProduct + bias`: includes user/item bias terms
  - `nn.Parameter`-based Embedding: manual implementation of latent matrices
  - `CollabNN`: shallow MLP with non-linear interaction
  - `Deep CollabNN`: 2-hidden-layer MLP with dropout and weight decay

- **Tools & Frameworks**:
  - `fastai.collab` (for efficient model prototyping)
  - `PyTorch` (as backend)
  - `Gradio` (for live model serving)
  - `Hugging Face Spaces` (for public web demo)

---

## 🚀 How to Run

### ▶️ Local Execution 

1. **Setup environment**
   ```bash
   conda create -n fastai python=3.11
   conda activate fastai
   pip install -U fastai gradio pandas torch

2.	Run the app
```bash
python app.py

```
3.	Launch the interface -> Gradio will provide a localhost URL where you can test the model interactively.

---

## 🌐 How to Deploy in Hugging Face Spaces
1.	Upload the following files to your Space:

- app.py: main Gradio script
- export.pkl: fastai-trained model file
- ratings.csv: dataset with user id, movie id, and title
- requirements.txt: specify compatible dependencies
	
 3.	Space Runtime:
Select Gradio (Python) as the app type.
	
 4.	Note on Compatibility:
	•	The exported model relies on fastai<2.8.0 due to deprecation of fastcore.transform.
Include this in your requirements.txt:
```bash
fastai==2.7.12
```

---

 ## 🛠️ Challenges & Solutions

| Issue | Description | Solution |
|-------|-------------|----------|
| ❌ `httpcore` module not found | Occurred due to partial installation of Gradio dependencies | Installed with `pip install -U "gradio[all]"` |
| ❌ `load_learner` fails on HF Spaces | FastAI model depended on deprecated `fastcore.transform` | Fixed by pinning `fastai==2.7.12` |
| ❌ `IndexError` during inference | User/item IDs were mapped incorrectly | Used `.dls.classes['user id'].o2i` for correct embedding lookup |
| 🔁 MPS vs CPU prediction inconsistency | Local training used Apple MPS (GPU), inference ran on CPU | Ensured ID mapping and data processing consistency across environments |



---

## 📈 Performance Snapshot 📙 0063_movie_recommend_CF.ipynb

| Model Variant        | RMSE   | MAE   |
|----------------------|--------|-------|
| DotProduct           | 1.05+  | 0.83  |
| + y_range            | 0.96   | 0.76  |
| + bias               | 0.95   | 0.75  |
| nn.Parameter         | 0.92   | 0.73  |
| CollabNN             | 0.916  | 0.724 |
| Deep MLP (2h)        | 0.927  | 0.725 |

---

## 📎 Demo

Try the live demo here (Gradio UI hosted on Hugging Face):
👉 Demo Link：https://huggingface.co/spaces/apple9855/movie-recommend 

<img width="1383" alt="image" src="https://github.com/user-attachments/assets/c3edc8b9-ca16-4a4a-a452-1a273d1030e9" />

---

## 👩‍💻 Author

This project was built as part of a deep learning portfolio, demonstrating practical skills in:

	•	Efficient model prototyping with fastai
	•	Data preprocessing for tabular + categorical embeddings
	•	Applying deep collaborative filtering models
	•	Model deployment via Gradio + Hugging Face

Feel free to reach out or open an issue for questions!


