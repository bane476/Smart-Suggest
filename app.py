import os
from flask import Flask, request, render_template
from dotenv import load_dotenv

load_dotenv()
import pandas as pd
import random
import numpy as np
from flask_sqlalchemy import SQLAlchemy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(_name_)

# load files
trending_products = pd.read_csv("models/trending_products.csv")
train_data = pd.read_csv("models/clean_data.csv")

# database configuration
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "a_fallback_secret_key_for_development_only")
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL")
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# Define your model class for the 'signup' table
class Signup(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)

# Define your model class for the 'signup' table
class Signin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), nullable=False)
    password = db.Column(db.String(100), nullable=False)


# Recommendations functions
# Function to truncate product name
def truncate(text, length):
    if len(text) > length:
        return text[:length] + "..."
    else:
        return text


def content_based_recommendations(train_data, item_name, top_n=10):
    # Check if any item name contains the search query as a whole word
    matching_items = train_data[train_data['Name'].str.contains(r'\b' + item_name + r'\b', case=False, na=False)]
    if matching_items.empty:
        print(f"No items matching '{item_name}' found in the training data.")
        return pd.DataFrame()

    # Use the first matching item as the reference for recommendations
    item_name = matching_items.iloc[0]['Name']

    # Create a TF-IDF vectorizer for item descriptions
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Apply TF-IDF vectorization to item descriptions
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])

    # Calculate cosine similarity between items based on descriptions
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)

    # Find the index of the item
    item_index = train_data[train_data['Name'] == item_name].index[0]

    # Get the cosine similarity scores for the item
    similar_items = list(enumerate(cosine_similarities_content[item_index]))

    # Sort similar items by similarity score in descending order
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)

    # Get the top N most similar items (excluding the item itself)
    top_similar_items = similar_items[1:top_n+1]

    # Get the indices of the top similar items
    recommended_item_indices = [x[0] for x in top_similar_items]

    # Get the details of the top similar items
    recommended_items_details = train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]

    return recommended_items_details

def collaborative_filtering_recommendations(train_data, target_user_id, top_n=10):
    # Check if required columns exist
    if 'ID' not in train_data.columns or 'ProdID' not in train_data.columns or 'Rating' not in train_data.columns:
        print("Required columns for collaborative filtering are missing.")
        return pd.DataFrame()

    user_item_matrix = train_data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0)
    if target_user_id not in user_item_matrix.index:
        print(f"User ID '{target_user_id}' not found in the data.")
        # Return empty dataframe as user is not found
        return pd.DataFrame()

    user_similarity = cosine_similarity(user_item_matrix)
    target_user_index = user_item_matrix.index.get_loc(target_user_id)
    user_similarities = user_similarity[target_user_index]
    similar_users_indices = user_similarities.argsort()[::-1][1:]

    recommended_items = []
    for user_index in similar_users_indices:
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        # Check if target_user_index is valid before using it
        if target_user_index < len(user_item_matrix):
            not_rated_by_target_user = (rated_by_similar_user > 0) & (user_item_matrix.iloc[target_user_index] == 0)
            recommended_items.extend(user_item_matrix.columns[not_rated_by_target_user][:top_n])
    
    recommended_items = list(set(recommended_items))[:top_n]

    recommended_items_details = train_data[train_data['ProdID'].isin(recommended_items)][['Name', 'ReviewCount', 'Brand', 'ImageURL', 'Rating']]
    return recommended_items_details.head(top_n)

def hybrid_recommendations(train_data, target_user_id, item_name, top_n=10):
    content_based_rec = content_based_recommendations(train_data, item_name, top_n)
    collaborative_filtering_rec = collaborative_filtering_recommendations(train_data, target_user_id, top_n)

    # Check if both dataframes are not empty before concatenating
    if not content_based_rec.empty and not collaborative_filtering_rec.empty:
        hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec]).drop_duplicates()
    elif not content_based_rec.empty:
        hybrid_rec = content_based_rec
    elif not collaborative_filtering_rec.empty:
        hybrid_rec = collaborative_filtering_rec
    else:
        return pd.DataFrame()

    if not hybrid_rec.empty:
        hybrid_rec['ImageURL'] = hybrid_rec['ImageURL'].apply(lambda x: x.split(' | ')[0])

    return hybrid_rec.head(top_n)

# routes
# List of predefined image URLs
random_image_urls = [
    "static/img_1.png",
    "static/img_2.png",
    "static/img_3.png",
    "static/img_4.png",
    "static/img_5.png",
    "static/img_6.png",
    "static/img_7.png",
    "static/img_8.png",
]


@app.route("/")
def index():
    # Create a list of random image URLs for each product
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html',trending_products=trending_products.head(8),truncate = truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price = random.choice(price))

@app.route("/main")
def main():
    return render_template('main.html', content_based_rec=pd.DataFrame())

# routes
@app.route("/index")
def indexredirect():
    # Create a list of random image URLs for each product
    random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
    price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
    return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                           random_product_image_urls=random_product_image_urls,
                           random_price=random.choice(price))

@app.route("/signup", methods=['POST','GET'])
def signup():
    if request.method=='POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        new_signup = Signup(username=username, email=email, password=password)
        db.session.add(new_signup)
        db.session.commit()

        # Create a list of random image URLs for each product
        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                               random_product_image_urls=random_product_image_urls, random_price=random.choice(price),
                               signup_message='User signed up successfully!'
                               )

# Route for signup page
@app.route('/signin', methods=['POST', 'GET'])
def signin():
    if request.method == 'POST':
        username = request.form['signinUsername']
        password = request.form['signinPassword']
        new_signup = Signin(username=username,password=password)
        db.session.add(new_signup)
        db.session.commit()

        # Create a list of random image URLs for each product
        random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(trending_products))]
        price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
        return render_template('index.html', trending_products=trending_products.head(8), truncate=truncate,
                               random_product_image_urls=random_product_image_urls, random_price=random.choice(price),
                               signup_message='User signed in successfully!'
                               )
@app.route("/recommendations", methods=['POST', 'GET'])
def recommendations():
    if request.method == 'POST':
        prod = request.form.get('prod')

        # Check if a product name was entered
        if not prod:
            message = "Please enter a product name to get recommendations."
            return render_template('main.html', message=message, content_based_rec=pd.DataFrame())

        nbr_str = request.form.get('nbr')
        if nbr_str:
            nbr = int(nbr_str)
        else:
            nbr = 10  # Default value if not provided
        
        # Hardcoded user ID for demonstration purposes
        target_user_id = 4
        
        # Get hybrid recommendations
        hybrid_rec = hybrid_recommendations(train_data, target_user_id, prod, top_n=nbr)

        if hybrid_rec.empty:
            message = f"No recommendations available for '{prod}'."
            return render_template('main.html', message=message, content_based_rec=pd.DataFrame(), truncate=truncate)
        else:
            # Create a list of random image URLs for each recommended product
            random_product_image_urls = [random.choice(random_image_urls) for _ in range(len(hybrid_rec))]
            price = [40, 50, 60, 70, 100, 122, 106, 50, 30, 50]
            return render_template('main.html', content_based_rec=hybrid_rec, truncate=truncate,
                                   random_product_image_urls=random_product_image_urls,
                                   random_price=random.choice(price),
                                   search_query=prod)


if _name=='main_':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
