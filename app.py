from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import re

app = FastAPI(title="Customer Feedback Analysis API")
try:
    bow_vectorizer = joblib.load('models/bow_vectorizer.pkl')
    tfidf_vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    nb_model_bow = joblib.load('models/nb_model_bow.pkl')
    lr_intent_model = joblib.load('models/lr_intent_model.pkl')
    nmf_model = joblib.load('models/nmf_model.pkl')
except Exception as e:
    print(f"Error loading models: {e}. Make sure the .pkl files are in the directory.")

# --- 2. YOUR LOGIC & MAPPINGS ---
topic_mapping = {
    0: "Tea & General Food Taste Feedback",
    1: "Roman Urdu Complaints & Dissatisfaction",
    2: "Price & Value-for-Money Feedback",
    3: "Coffee Products & Roast Strength",
    4: "General Product & Service Quality (Roman Urdu)"
}

# (Add your actual preprocess_text logic here)
def preprocess_text(text):
    # Example basic cleanup - replace this with your actual preprocessing code
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) 
    return text

def analyze_customer_feedback(review_text):
    cleaned_text = preprocess_text(review_text)

    vectorized_bow = bow_vectorizer.transform([cleaned_text])
    vectorized_tfidf = tfidf_vectorizer.transform([cleaned_text])

    sentiment_num = nb_model_bow.predict(vectorized_bow)[0]
    sentiment_label = {0: "Negative", 1: "Neutral", 2: "Positive"}.get(sentiment_num, "Unknown")

    intent_label = lr_intent_model.predict(vectorized_bow)[0]

    topic_distribution = nmf_model.transform(vectorized_tfidf)
    dominant_topic_idx = np.argmax(topic_distribution)
    topic_label = topic_mapping.get(dominant_topic_idx, "Unknown Topic")

    return sentiment_label, intent_label, topic_label

# --- 3. API ENDPOINTS ---

# Data Validation Model
class ReviewRequest(BaseModel):
    text: str

# Endpoint 1: Health Check (Required for Assignment)
@app.get("/")
def health_check():
    return {"status": "Customer Feedback API is active and running."}

# Endpoint 2: Prediction (Required for Assignment)
@app.post("/predict")
def predict_feedback(request: ReviewRequest):
    if not request.text or request.text.strip() == "":
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
    
    try:
        sentiment, intent, topic = analyze_customer_feedback(request.text)
        
        return {
            "original_text": request.text,
            "sentiment": sentiment,
            "intent": intent,
            "topic": topic
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")