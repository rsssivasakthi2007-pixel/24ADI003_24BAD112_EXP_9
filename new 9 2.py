print("SIVASAKTHI S 24BAD112")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


zip_path = r"C:\Users\priya\Downloads\archive (31).zip"

with zipfile.ZipFile(zip_path) as z:
    ratings = pd.read_csv(
        z.open('ml-100k/u.data'),
        sep='\t',
        names=['userId', 'movieId', 'rating', 'timestamp']
    )

    movies = pd.read_csv(
        z.open('ml-100k/u.item'),
        sep='|',
        encoding='latin-1',
        names=[
            'movieId','title','release_date','video_release_date','IMDb_URL',
            'unknown','Action','Adventure','Animation','Children','Comedy',
            'Crime','Documentary','Drama','Fantasy','Film-Noir','Horror',
            'Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'
        ]
    )


train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

train_matrix = train_data.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)
test_matrix = test_data.pivot_table(index='movieId', columns='userId', values='rating').fillna(0)


item_similarity = cosine_similarity(train_matrix)
item_similarity_df = pd.DataFrame(item_similarity, index=train_matrix.index, columns=train_matrix.index)

def get_similar_items(movie_id, top_n=5):
    if movie_id not in item_similarity_df.index:
        return []
    similar_scores = item_similarity_df[movie_id].sort_values(ascending=False)
    return similar_scores.iloc[1:top_n+1]

def recommend_movies(user_id, top_n=5):
    if user_id not in train_matrix.columns:
        return []
    user_ratings = train_matrix[user_id]
    watched_movies = user_ratings[user_ratings > 0].index
    scores = {}
    for movie in watched_movies:
        similar_items = item_similarity_df[movie]
        for sim_movie, score in similar_items.items():
            if sim_movie not in watched_movies:
                scores[sim_movie] = scores.get(sim_movie, 0) + score
    recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in recommended[:top_n]]


user_similarity = cosine_similarity(train_matrix.T)
user_similarity_df = pd.DataFrame(user_similarity, index=train_matrix.columns, columns=train_matrix.columns)

def user_based_recommend(user_id, top_n=5):
    if user_id not in user_similarity_df.index:
        return []
    sim_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]
    scores = {}
    for sim_user in sim_users.index:
        sim_score = sim_users[sim_user]
        user_ratings = train_matrix[sim_user]
        for movie, rating in user_ratings.items():
            if rating > 0:
                scores[movie] = scores.get(movie, 0) + sim_score * rating
    recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [x[0] for x in recommended[:top_n]]


def precision_at_k(user_id, k=5):
    recommended = recommend_movies(user_id, k)
    if user_id not in test_matrix.columns:
        return 0
    relevant = test_matrix[user_id]
    relevant_items = relevant[relevant > 3].index.tolist()
    if len(relevant_items) == 0:
        return 0
    hits = len(set(recommended) & set(relevant_items))
    return hits / k

def precision_at_k_user(user_id, k=5):
    recommended = user_based_recommend(user_id, k)
    if user_id not in test_matrix.columns:
        return 0
    relevant = test_matrix[user_id]
    relevant_items = relevant[relevant > 3].index.tolist()
    if len(relevant_items) == 0:
        return 0
    hits = len(set(recommended) & set(relevant_items))
    return hits / k

model = DecisionTreeRegressor()
model.fit(train_data[['userId', 'movieId']], train_data['rating'])

y_true = test_data['rating']
y_pred_tree = model.predict(test_data[['userId', 'movieId']])
rmse_tree = np.sqrt(mean_squared_error(y_true, y_pred_tree))

user_id = 1
item_recommendations = recommend_movies(user_id)
user_recommendations = user_based_recommend(user_id)


def get_movie_names(movie_ids):
    return movies[movies['movieId'].isin(movie_ids)]['title'].tolist()


plt.figure(figsize=(8,6))
sns.heatmap(item_similarity_df.iloc[:20, :20])
plt.title("Item Similarity Heatmap")
plt.show()

similar_items = get_similar_items(1, 5)

plt.figure()
similar_items.plot(kind='bar')
plt.title("Top Similar Items for Movie 1")
plt.xlabel("Movie ID")
plt.ylabel("Similarity Score")
plt.show()

labels = ['Item-Based', 'User-Based']
values = [precision_at_k(user_id, 5), precision_at_k_user(user_id, 5)]

plt.figure()
plt.bar(labels, values)
plt.title("Recommendation Comparison")
plt.ylabel("Precision@5")
plt.show()


print("Recommended Movies (Item-Based):", get_movie_names(item_recommendations))
print("Recommended Movies (User-Based):", get_movie_names(user_recommendations))
print("RMSE (Decision Tree):", rmse_tree)
print("Precision@5 (Item-Based):", precision_at_k(user_id, 5))
print("Precision@5 (User-Based):", precision_at_k_user(user_id, 5))

print("\n========== ANALYSIS ==========")

print("\n1. Accuracy:")
print("- Item-based filtering gives stable recommendations.")
print("- User-based filtering depends on user similarity.")

print("\n2. Popular vs Niche:")
print("- Popular items have more interactions → better similarity.")
print("- Niche items suffer due to sparse data.")

print("\n3. Scalability:")
print("- Item-based filtering is scalable.")
print("- Suitable for large number of users.")

print("\n========== END ==========")
