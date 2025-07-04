

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# --- Recommendation Functions (copied from app.py) ---

def content_based_recommendations(train_data, item_name, top_n=10):
    # Escape item_name to handle special regex characters
    safe_item_name = re.escape(item_name)
    matching_items = train_data[train_data['Name'].str.contains(r'\b' + safe_item_name + r'\b', case=False, na=False)]
    if matching_items.empty:
        return pd.DataFrame()

    item_name = matching_items.iloc[0]['Name']
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)
    
    # Find the integer position of the item in the dataframe
    item_index_label = train_data[train_data['Name'] == item_name].index[0]
    item_index_pos = train_data.index.get_loc(item_index_label)

    similar_items = list(enumerate(cosine_similarities_content[item_index_pos]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    top_similar_items = similar_items[1:top_n+1]
    recommended_item_indices = [x[0] for x in top_similar_items]
    return train_data.iloc[recommended_item_indices]

def collaborative_filtering_recommendations(train_data, target_user_id, top_n=10):
    if 'ID' not in train_data.columns or 'ProdID' not in train_data.columns or 'Rating' not in train_data.columns:
        return pd.DataFrame()

    user_item_matrix = train_data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0)
    if target_user_id not in user_item_matrix.index:
        return pd.DataFrame()

    user_similarity = cosine_similarity(user_item_matrix)
    target_user_index = user_item_matrix.index.get_loc(target_user_id)
    user_similarities = user_similarity[target_user_index]
    similar_users_indices = user_similarities.argsort()[::-1][1:]

    recommended_items = []
    for user_index in similar_users_indices:
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        if target_user_index < len(user_item_matrix):
            not_rated_by_target_user = (rated_by_similar_user > 0) & (user_item_matrix.iloc[target_user_index] == 0)
            recommended_items.extend(user_item_matrix.columns[not_rated_by_target_user][:top_n])
    
    recommended_items = list(set(recommended_items))[:top_n]
    return train_data[train_data['ProdID'].isin(recommended_items)]

def hybrid_recommendations(train_data, target_user_id, item_name, top_n=10):
    content_based_rec = content_based_recommendations(train_data, item_name, top_n)
    collaborative_filtering_rec = collaborative_filtering_recommendations(train_data, target_user_id, top_n)

    if not content_based_rec.empty and not collaborative_filtering_rec.empty:
        hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec]).drop_duplicates()
    elif not content_based_rec.empty:
        hybrid_rec = content_based_rec
    elif not collaborative_filtering_rec.empty:
        hybrid_rec = collaborative_filtering_rec
    else:
        return pd.DataFrame()

    return hybrid_rec.head(top_n)

# --- Evaluation Metrics ---

def precision_at_k(recommended_items, relevant_items, k):
    recommended_k = recommended_items.head(k)
    relevant_set = set(relevant_items['ProdID'])
    recommended_set = set(recommended_k['ProdID'])
    
    true_positives = len(recommended_set.intersection(relevant_set))
    return true_positives / k if k > 0 else 0

def recall_at_k(recommended_items, relevant_items, k):
    recommended_k = recommended_items.head(k)
    relevant_set = set(relevant_items['ProdID'])
    recommended_set = set(recommended_k['ProdID'])
    
    true_positives = len(recommended_set.intersection(relevant_set))
    return true_positives / len(relevant_set) if len(relevant_set) > 0 else 0

def ndcg_at_k(recommended_items, relevant_items, k):
    recommended_k = recommended_items.head(k).reset_index(drop=True)
    relevant_set = set(relevant_items['ProdID'])
    
    dcg = 0
    for i, row in recommended_k.iterrows():
        if row['ProdID'] in relevant_set:
            dcg += 1 / np.log2(i + 2) # i is now the rank 0, 1, 2...
            
    idcg = 0
    for i in range(min(k, len(relevant_set))):
        idcg += 1 / np.log2(i + 2)
        
    return dcg / idcg if idcg > 0 else 0

# --- Evaluation ---

def evaluate_recommenders(data, k=10):
    print("Starting evaluation...")
    # Split data into 80% train and 20% test
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    print(f"Data split: {len(train_df)} training samples, {len(test_df)} testing samples.")

    # Get unique users in the test set
    test_users = test_df['ID'].unique()
    print(f"Found {len(test_users)} unique users in the test set.")
    if not len(test_users):
        print("No test users found. Aborting evaluation.")
        return {}

    metrics = {
        'content_based': {'precision': [], 'recall': [], 'ndcg': []},
        'collaborative': {'precision': [], 'recall': [], 'ndcg': []},
        'hybrid': {'precision': [], 'recall': [], 'ndcg': []}
    }

    for i, user_id in enumerate(test_users):
        if (i + 1) % 50 == 0: # Print progress every 50 users
             print(f"Processing user {i + 1}/{len(test_users)}...")

        # Get user's interactions from the test set (ground truth)
        relevant_items = test_df[test_df['ID'] == user_id]
        
        # We need an item to seed the content-based and hybrid recommenders
        # We'll pick a random item that the user has interacted with in the training set
        user_train_items = train_df[train_df['ID'] == user_id]
        if user_train_items.empty:
            continue
        
        # Seed item for content-based and hybrid
        seed_item_name = user_train_items.iloc[0]['Name']

        # Get recommendations
        cb_rec = content_based_recommendations(train_df, seed_item_name, top_n=k)
        cf_rec = collaborative_filtering_recommendations(train_df, user_id, top_n=k)
        hybrid_rec = hybrid_recommendations(train_df, user_id, seed_item_name, top_n=k)

        # Evaluate Content-Based
        if not cb_rec.empty:
            metrics['content_based']['precision'].append(precision_at_k(cb_rec, relevant_items, k))
            metrics['content_based']['recall'].append(recall_at_k(cb_rec, relevant_items, k))
            metrics['content_based']['ndcg'].append(ndcg_at_k(cb_rec, relevant_items, k))

        # Evaluate Collaborative
        if not cf_rec.empty:
            metrics['collaborative']['precision'].append(precision_at_k(cf_rec, relevant_items, k))
            metrics['collaborative']['recall'].append(recall_at_k(cf_rec, relevant_items, k))
            metrics['collaborative']['ndcg'].append(ndcg_at_k(cf_rec, relevant_items, k))

        # Evaluate Hybrid
        if not hybrid_rec.empty:
            metrics['hybrid']['precision'].append(precision_at_k(hybrid_rec, relevant_items, k))
            metrics['hybrid']['recall'].append(recall_at_k(hybrid_rec, relevant_items, k))
            metrics['hybrid']['ndcg'].append(ndcg_at_k(hybrid_rec, relevant_items, k))

    # Calculate average metrics
    avg_metrics = {}
    for rec_type, metric_values in metrics.items():
        avg_metrics[rec_type] = {
            'Precision@' + str(k): np.mean(metric_values['precision']),
            'Recall@' + str(k): np.mean(metric_values['recall']),
            'nDCG@' + str(k): np.mean(metric_values['ndcg'])
        }
        
    return avg_metrics

if __name__ == '__main__':
    print("Loading data...")
    # Load the dataset
    train_data = pd.read_csv("models/clean_data.csv")
    print("Data loaded successfully.")
    results = evaluate_recommenders(train_data, k=10)
    print("Evaluation complete. Final results:")
    for recommender, metrics in results.items():
        print(f"\n--- {recommender.replace('_', ' ').title()} ---")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
