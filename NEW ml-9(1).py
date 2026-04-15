print("SIVASAKTHI 24BAD112")
import pandas as pd
import numpy as np
import zipfile
import os

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

zip_path = r"C:\Users\priya\Downloads\archive (29).zip"
extract_path = "movielens_data"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
print(" Extraction completed!")

ratings_path, movies_path = None, None

for root, dirs, files in os.walk(extract_path):   
    for file in files:
        if file == "u.data":
            ratings_path = os.path.join(root, file)
        if file == "u.item":
            movies_path = os.path.join(root, file)
        if file == "ratings.csv":
            ratings_path = os.path.join(root, file)

print("Ratings File:", ratings_path)
print("Movies File:", movies_path)

if ratings_path.endswith("u.data"):
    ratings = pd.read_csv(ratings_path, sep='\t',
                          names=['user_id', 'movie_id', 'rating', 'timestamp'])
else:
    ratings = pd.read_csv(ratings_path)
    ratings.columns = ['user_id', 'movie_id', 'rating', 'timestamp']
    
print("\n Dataset Info:")
print(ratings.info())

print("\nFirst 5 rows:")
print(ratings.head())

print("\nMissing values:")
print(ratings.isnull().sum())

if movies_path:
    movies = pd.read_csv(movies_path, sep='|', encoding='latin-1', header=None)
    movies = movies[[0, 1]]
    movies.columns = ['movie_id', 'title']
else:
    movies = pd.DataFrame({'movie_id': ratings['movie_id'].unique(),
                           'title': ratings['movie_id'].unique()})

train, test = train_test_split(ratings, test_size=0.2, random_state=42)

train_matrix = train.pivot_table(index='user_id',
                                 columns='movie_id',
                                 values='rating')

user_mean = train_matrix.mean(axis=1)
train_centered = train_matrix.sub(user_mean, axis=0)
train_filled = train_centered.fillna(0)

user_similarity = cosine_similarity(train_filled)

user_similarity_df = pd.DataFrame(user_similarity,
                                 index=train_filled.index,
                                 columns=train_filled.index)

def predict_single(user, movie, k=10):
    if user not in train_matrix.index:
        return train['rating'].mean()

    if movie not in train_matrix.columns:
        return user_mean[user]

    similar_users = user_similarity_df[user].sort_values(ascending=False)[1:k+1]

    num, den = 0, 0

    for sim_user, score in similar_users.items():
        rating = train_matrix.loc[sim_user, movie]
        if not np.isnan(rating):
            num += score * (rating - user_mean[sim_user])
            den += abs(score)

    if den == 0:
        return user_mean[user]

    return user_mean[user] + (num / den)

y_true, y_pred = [], []

for _, row in test.iterrows():
    pred = predict_single(row['user_id'], row['movie_id'])
    y_true.append(row['rating'])
    y_pred.append(pred)

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae = mean_absolute_error(y_true, y_pred)

print("\n RMSE:", rmse)
print(" MAE:", mae)

def recommend_movies(user_id, n=5):
    if user_id not in train_matrix.index:
        print(" User not found!")
        return None, [], []

    all_movies = train_matrix.columns
    watched = train_matrix.loc[user_id].dropna().index

    predictions = {}

    for movie in all_movies:
        if movie not in watched:
            predictions[movie] = predict_single(user_id, movie)

    top_movies = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:n]

    movie_ids = [i[0] for i in top_movies]
    scores = [i[1] for i in top_movies]

    result = movies[movies['movie_id'].isin(movie_ids)]

    return result, movie_ids, scores   

user_id = int(input("\nEnter User ID: "))

print(f"\n Recommended Movies for User {user_id}:\n")
recommended, movie_ids, scores = recommend_movies(user_id)

print(f"\n Recommended Movies for User {user_id}:\n")
print(recommended)

titles = []

for mid in movie_ids:
    title = movies[movies['movie_id'] == mid]['title'].values
    if len(title) > 0:
        titles.append(title[0])
    else:
        titles.append(str(mid))

plt.figure(figsize=(8,5))
plt.bar(titles, scores)
plt.title(f"Top Recommended Movies for User {user_id}")
plt.xlabel("Movie Title")
plt.ylabel("Predicted Rating")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print(f"\n Top similar users for User {user_id}:")
similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:6]

for u, score in similar_users.items():
    print(f"User {u} → Similarity: {score:.2f}")

print("\n Similarity Matrix Meaning:")
print("1.0   → Identical users")
print("0.7 – 1.0 → Highly similar")
print("0.4 – 0.7 → Moderately similar")
print("0.1 – 0.4 → Slight similarity")
print("0     → No similarity")


plt.figure(figsize=(12, 8))

heatmap1 = sns.heatmap(
    train_matrix.fillna(0).iloc[:20, :20],
    cmap='coolwarm',
    linewidths=0.5,
    linecolor='black',
    cbar=True
)

cbar1 = heatmap1.collections[0].colorbar
cbar1.set_label("Rating Value")

plt.title("User-Item Matrix Heatmap")
plt.xlabel("Movie ID")
plt.ylabel("User ID")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))

heatmap2 = sns.heatmap(
    user_similarity_df.iloc[:20, :20],
    cmap='viridis',
    linewidths=0.5,
    linecolor='black',
    cbar=True
)

cbar2 = heatmap2.collections[0].colorbar
cbar2.set_label("Similarity Score")

plt.title("User Similarity Matrix")
plt.xlabel("Users")
plt.ylabel("Users")

plt.tight_layout()
plt.show()

print("\n Heatmap Color Meaning:")
print("Blue → Low ratings")
print("White → Medium ratings")
print("Red → High ratings")

print("\n  IMPORTANT:")
print(" Similarity matrix DOES NOT change when user changes")
print(" Only selected user row changes → recommendations change")
