import gradio as gr
import pandas as pd
import torch
from fastai.collab import load_learner
from fastai.torch_core import tensor

# ✅ Load model & data
learn = load_learner('export.pkl')
ratings = pd.read_csv('ratings.csv')

# 👇 封闭环境，使用外部变量
def Rec_top_n(user_id, n=5):
    # user 和 item 的映射表
    user_map = learn.dls.classes['user id']
    item_map = learn.dls.classes['title']

    if user_id not in user_map.o2i:
        return f"❌ User ID {user_id} not found in training data."

    # 获取用户已看和未看电影（按 title）
    seen_titles = ratings[ratings['user id'] == user_id]['title'].values
    all_titles = ratings['title'].unique()
    unseen_titles = [t for t in all_titles if t not in seen_titles and t in item_map.o2i]

    if len(unseen_titles) == 0:
        return f"⚠️ No unseen movies for user {user_id}"

    user_idx = user_map.o2i[user_id]
    item_idxs = [item_map.o2i[t] for t in unseen_titles]

    device = learn.model.i_weight.weight.device
    x = tensor([[user_idx, i] for i in item_idxs]).to(device)
    preds = learn.model(x).squeeze()

    top_idxs = preds.argsort(descending=True)[:n]
    top_titles = [unseen_titles[i] for i in top_idxs]

    return "\n".join([f"{i+1}. {title}" for i, title in enumerate(top_titles)])

demo = gr.Interface(
    fn=Rec_top_n,
    inputs=gr.Number(label="User ID (Hint: 1~943)"),
    outputs=gr.Textbox(label="Top Movie Recommendations"),
    title="🎬 Movie Recommender (MovieLens 100K)",
    description="Enter a User ID to get top-N movie recommendations based on collaborative filtering."
)

demo.launch()