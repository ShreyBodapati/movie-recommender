# 🎬 Movie Recommender

A personalized movie recommendation app powered by collaborative filtering and AI-generated explanations.

🔗 **[Live Demo](https://shrey-movie-recommender.streamlit.app)**

---

## How It Works

1. User enters a movie they love
2. Collaborative filtering finds similar movies based on 100,000 real user ratings
3. An AI model explains *why* those movies were recommended in natural language

---

## Key Features

- **Collaborative filtering** using cosine similarity on the MovieLens 100K dataset
- **AI explanations** via HuggingFace Inference API
- **Movie search browser** to explore available titles
- Adjustable number of recommendations (3–10)

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.10-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red)
![HuggingFace](https://img.shields.io/badge/HuggingFace-API-yellow)

| Tool | Purpose |
|------|---------|
| Pandas | Data loading and matrix construction |
| scikit-learn | Cosine similarity for collaborative filtering |
| Streamlit | App framework and deployment |
| HuggingFace API | AI-generated recommendation explanations |

---

## Project Structure

```
movie-recommender/
├── data/
│   └── raw/ml-100k/       ← MovieLens 100K dataset
├── app/
│   └── app.py             ← Streamlit app
├── requirements.txt
└── README.md
```

---

## Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommender.git
cd movie-recommender
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Download MovieLens 100K from https://grouplens.org/datasets/movielens/100k/
# Place ml-100k folder in data/raw/
streamlit run app/app.py
```

---

## Dataset

MovieLens 100K — 100,000 ratings from 943 users on 1,682 movies collected by GroupLens Research.

Source: [grouplens.org/datasets/movielens/100k](https://grouplens.org/datasets/movielens/100k)

---

## What I Learned

- How collaborative filtering works using user-movie rating matrices
- Computing cosine similarity at scale with scikit-learn
- Integrating a free LLM API to add natural language explanations to ML outputs
- Building and deploying a full end-to-end ML application with Streamlit
