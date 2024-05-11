import joblib

def save_object(obj, filename):
    joblib.dump(obj, filename)

def load_object(filename):
    return joblib.load(filename)
