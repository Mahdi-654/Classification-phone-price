# models/model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

def load_and_train_model(train_path):
    # Charger les données
    data = pd.read_csv(train_path)
    X = data.drop(columns=['price_range'])
    y = data['price_range']

    # Diviser les données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entraîner le modèle
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Sauvegarder le modèle
    joblib.dump(model, 'models/model.pkl')

    return model

def predict_with_model(test_path):
    # Charger le modèle sauvegardé
    model = joblib.load('models/model.pkl')

    # Charger les données de test
    test_data = pd.read_csv(test_path)

    # Faire des prédictions
    predictions = model.predict(test_data)
    return predictions
