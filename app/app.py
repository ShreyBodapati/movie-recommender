import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# --- LOAD DATA ---
@st.cache_data
def load_data():
    # Load ratings — each row is one user rating one movie
    ratings = pd.read_csv(
        'data/raw/ml-100k/u.data',
        sep='\t',
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    )

    # Load movie titles
    movies = pd.read_csv(
        'data/raw/ml-100k/u.item',
        sep='|',
        encoding='latin-1',
        usecols=[0, 1],
        names=['movie_id', 'title']
    )

    # Create user-movie matrix — rows are users, columns are movies, values are ratings
    matrix = ratings.pivot_table(index='user_id', columns='movie_id', values='rating')

    # Fill NaN with 0 — needed for cosine similarity calculation
    matrix_filled = matrix.fillna(0)

    # Compute movie-movie similarity
    movie_similarity = cosine_similarity(matrix_filled.T)
    similarity_df = pd.DataFrame(
        movie_similarity,
        index=matrix.columns,
        columns=matrix.columns
    )

    return movies, similarity_df

movies, similarity_df = load_data()

# --- RECOMMENDATION FUNCTION ---
def get_recommendations(movie_title, n=5):
    match = movies[movies['title'].str.lower() == movie_title.lower()]

    if match.empty:
        match = movies[movies['title'].str.lower().str.contains(movie_title.lower())]

    if match.empty:
        return None, f"Could not find '{movie_title}' in the database"

    movie_id = match.iloc[0]['movie_id']

    if movie_id not in similarity_df.columns:
        return None, f"No rating data available for '{movie_title}'"

    similar_scores = similarity_df[movie_id].sort_values(ascending=False)
    similar_scores = similar_scores.drop(movie_id, errors='ignore')
    top_ids = similar_scores.head(n).index.tolist()

    recommended = movies[movies['movie_id'].isin(top_ids)][['movie_id', 'title']].copy()
    recommended['similarity'] = recommended['movie_id'].map(similar_scores)
    recommended = recommended.sort_values('similarity', ascending=False)

    return recommended, None

# --- AI EXPLANATION FUNCTION ---
def get_ai_explanation(input_movie, recommended_movies):
    movie_list = ', '.join(recommended_movies['title'].tolist())

    prompt = f"""You are a movie expert. A user liked "{input_movie}" and our recommendation system suggested: {movie_list}.

In 2-3 sentences, explain what these movies have in common with "{input_movie}" and why a fan would enjoy them. Be specific about themes, style, or genre."""

    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            json={"inputs": prompt, "parameters": {"max_new_tokens": 150}},
            timeout=15
        )

        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated = result[0].get('generated_text', '')
                explanation = generated.replace(prompt, '').strip()
                return explanation if explanation else "These films share similar themes and storytelling styles that fans of your selection tend to enjoy."
        return "These films share similar themes and storytelling styles that fans of your selection tend to enjoy."

    except Exception:
        return "These films share similar themes and storytelling styles that fans of your selection tend to enjoy."

# --- UI ---
st.title("🎬 Movie Recommender")
st.markdown("Type a movie you love and get personalized recommendations powered by collaborative filtering + AI.")

st.divider()

col1, col2 = st.columns([2, 1])

with col1:
    movie_input = st.text_input(
        "Enter a movie title",
        placeholder="e.g. Toy Story, Star Wars, Fargo"
    )

with col2:
    num_recs = st.slider("Number of recommendations", min_value=3, max_value=10, value=5)

if st.button("Get Recommendations", type="primary"):
    if not movie_input:
        st.warning("Please enter a movie title first.")
    else:
        with st.spinner("Finding similar movies..."):
            recommendations, error = get_recommendations(movie_input, n=num_recs)

        if error:
            st.error(error)
            st.info("Try searching for a popular movie like 'Toy Story', 'Star Wars', or 'Fargo'")
        else:
            st.success(f"Found {len(recommendations)} recommendations for **{movie_input}**")

            st.subheader("Recommended Movies")
            for _, row in recommendations.iterrows():
                similarity_pct = round(row['similarity'] * 100, 1)
                st.markdown(f"**{row['title']}** — {similarity_pct}% match")

            st.divider()

            st.subheader("Why these movies?")
            with st.spinner("Generating AI explanation..."):
                explanation = get_ai_explanation(movie_input, recommendations)
            st.markdown(explanation)

st.divider()

with st.expander("Browse available movies"):
    search = st.text_input("Search movies in database", placeholder="Type to search...")
    if search:
        results = movies[movies['title'].str.lower().str.contains(search.lower())]
        st.dataframe(results[['title']].head(20), use_container_width=True)
    else:
        st.dataframe(movies[['title']].head(50), use_container_width=True)