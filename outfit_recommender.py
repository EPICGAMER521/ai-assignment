from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import random

app = Flask(__name__)

# Sample dataset (replace with a real dataset from Kaggle or other sources)
data = {
    'item_id': [1, 2, 3, 4, 5, 6, 7, 8],
    'type': ['top', 'bottom', 'top', 'outerwear', 'footwear', 'accessory', 'top', 'bottom'],
    'color': ['red', 'blue', 'black', 'green', 'black', 'silver', 'white', 'black'],
    'material': ['cotton', 'denim', 'cotton', 'wool', 'leather', 'metal', 'silk', 'cotton']
}
df = pd.DataFrame(data)

# Preprocess the dataset
def preprocess_data(df):
    # Encode categorical variables
    df_encoded = pd.get_dummies(df, columns=['type', 'color', 'material'])
    # Normalize numerical features (if any, for future scalability)
    scaler = StandardScaler()
    df_encoded = pd.DataFrame(scaler.fit_transform(df_encoded), columns=df_encoded.columns)
    return df_encoded

# Compute similarity matrix
def compute_similarity_matrix(df_encoded):
    return cosine_similarity(df_encoded)

# Recommend outfits based on input item
def recommend_outfits(item_id, df, similarity_matrix, top_n=3):
    item_idx = df.index[df['item_id'] == item_id].tolist()[0]
    sim_scores = list(enumerate(similarity_matrix[item_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]  # Exclude the item itself
    outfit_indices = [i[0] for i in sim_scores]
    return df.iloc[outfit_indices][['item_id', 'type', 'color', 'material']].to_dict('records')

# Preprocess data and compute similarity matrix
df_encoded = preprocess_data(df)
similarity_matrix = compute_similarity_matrix(df_encoded)

@app.route('/')
def index():
    return render_template('index.html', items=df.to_dict('records'))

@app.route('/recommend', methods=['POST'])
def recommend():
    item_id = int(request.form['item_id'])
    recommendations = recommend_outfits(item_id, df, similarity_matrix)
    return render_template('index.html', items=df.to_dict('records'), recommendations=recommendations, selected_item_id=item_id)

if __name__ == '__main__':
    app.run(debug=True)