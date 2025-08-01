import joblib
from backend.utils.preprocess import preprocess_text

# Load saved model and vectorizer
model = joblib.load("backend/sarcasm/sarcasm_model.pkl")
vectorizer = joblib.load("backend/sarcasm/sarcasm_vectorizer.pkl")

def detect_sarcasm(text):
    """
    Predict if the given text is sarcastic.
    Returns:
      - is_sarcastic: True/False
      - weightage: Confidence in percentage
    """
    cleaned_text = preprocess_text(text)
    features = vectorizer.transform([cleaned_text])
    
    # Get probability score
    probas = model.predict_proba(features)[0]  # [0]: get first result from list

    sarcasm_confidence = probas[1]  # 1 is sarcastic class
    is_sarcastic = sarcasm_confidence > 0.5

    return is_sarcastic, round(sarcasm_confidence * 100, 2)  # True/False, percentage

# import joblib
# from backend.utils.preprocess import preprocess_text

# # Load the saved sarcasm detection model
# model = joblib.load("backend/sarcasm/sarcasm_model.pkl")
# vectorizer = joblib.load("backend/sarcasm/sarcasm_vectorizer.pkl")

# def detect_sarcasm(text):
#     """
#     Predict if the given text is sarcastic or not.
#     Returns True (sarcasm) or False (not sarcasm).
#     """
#     cleaned_text = preprocess_text(text)
#     features = vectorizer.transform([cleaned_text])
#     prediction = model.predict(features)

#     return bool(prediction[0])  # True if sarcastic
