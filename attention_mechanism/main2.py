# main.py

from data_preprocessing import TextPreprocessor
import train_evaluate as te
import pandas as pd

def main():
    data = pd.read_csv('../dataset/Suicide_Detection.csv')
    preprocessor = TextPreprocessor()
    X = preprocessor.preprocess_texts(data['text'])
    y = data['class'].apply(lambda x: 1 if x == "suicide" else 0).values

    # Split data
    X_train, X_test, y_train, y_test = te.train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Tune and get the best model
    best_model = te.tune_model(X_train, y_train, input_dim=5000+1, max_length=200)
    
    # Train the final model using the best hyperparameters
    history = te.train_final_model(X_train, y_train, X_test, y_test, best_model)
    print("Model training complete.")

if __name__ == "__main__":
    main()
