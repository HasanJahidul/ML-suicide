import joblib
import data_preprocessing as dp

def load_model(filename):
    return joblib.load(filename)

def make_prediction(model, vectorizer, text):
    cleaned_text = dp.clean_text(text)  # Assuming clean_text is imported
    tfidf_features = vectorizer.transform([cleaned_text])
    return model.predict(tfidf_features)
