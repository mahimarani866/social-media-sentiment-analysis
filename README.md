# Social Media Sentiment Analysis (Python & NLP)

Simple end-to-end sentiment analysis project on ~12K synthetic social media posts across multiple brands and platforms.

## What I Did
- Generated a synthetic dataset of social posts (brand, platform, text, sentiment)
- Built an NLP pipeline with TF-IDF + Logistic Regression
- Classified posts into Positive / Negative / Neutral
- Created a brand-level sentiment summary for 4 brands
- Added a small insights script to print which brand has better sentiment

## Tech Stack
- Python
- pandas
- scikit-learn
- TF-IDF
- Logistic Regression

## 🗂 Files
- `generate_social_data.py` – creates `data/social_posts_raw.csv`
- `train_sentiment_model.py` – trains model and saves:
  - `models/sentiment_pipeline.joblib`
  - `data/brand_sentiment_summary.csv`
- `brand_insights.py` – prints simple brand sentiment insights



