import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Example data - replace with real dataset
items = pd.DataFrame({
    'item_id': [1,2,3,4],
    'title': ['Movie A','Movie B','Movie C','Movie D'],
    'description': [
        'action adventure hero',
        'romantic drama love',
        'action thriller adventure',
        'romantic comedy light-hearted love'
    ]
})

# Content-based: TF-IDF on description
tf = TfidfVectorizer()
tfidf_matrix = tf.fit_transform(items['description'])
item_sim = cosine_similarity(tfidf_matrix)  # item-item similarity matrix

def content_recommend(item_id, top_n=3):
    idx = items.index[items['item_id'] == item_id].tolist()[0]
    sims = item_sim[idx]
    top_idx = np.argsort(sims)[::-1][1:top_n+1]  # skip itself
    return items.iloc[top_idx][['item_id','title']]

print("Content-based recommendations for item 1:")
print(content_recommend(1))

# Collaborative filtering (simple item-based using user ratings)
# Example ratings: rows = user, columns = item_id
ratings = pd.DataFrame({
    'user_id': [10,10,20,20,30],
    'item_id': [1,2,2,3,4],
    'rating': [5,4,5,4,3]
})
# Build user-item pivot
pivot = ratings.pivot_table(index='user_id', columns='item_id', values='rating').fillna(0)
# Item vectors
item_vectors = pivot.T.values  # shape (n_items, n_users)
item_sim_collab = cosine_similarity(item_vectors)

def predict_item_score(user_id, target_item_id):
    # item-based CF: weighted average of user's ratings on other items
    if target_item_id not in pivot.columns:
        return 0
    user_ratings = pivot.loc[user_id] if user_id in pivot.index else pd.Series(0, index=pivot.columns)
    target_idx = list(pivot.columns).index(target_item_id)
    sims = item_sim_collab[target_idx]
    # ignore items with zero rating by user
    rated_mask = user_ratings > 0
    if rated_mask.sum() == 0: return 0
    numer = (sims * user_ratings.values).sum()
    denom = np.abs(sims[rated_mask]).sum()
    return numer / denom if denom != 0 else 0

# Example predict
print("Predicted rating for user 10 on item 3:", predict_item_score(10, 3))

