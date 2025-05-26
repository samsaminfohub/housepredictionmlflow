import joblib
import numpy as np
import os

class HousingPredictor:
    def __init__(self, model_path="models/housing_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.load_model()
    
    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"Modèle non trouvé: {self.model_path}")
    
    def predict(self, features):
        if self.model is None:
            raise ValueError("Modèle non chargé")
        return self.model.predict(features)
    
    def predict_single(self, surface, rooms, bedrooms, age, location_score,
                      has_garden, has_parking, has_balcony, energy_class):
        features = np.array([[
            surface, rooms, bedrooms, age, location_score,
            int(has_garden), int(has_parking), int(has_balcony),
            energy_class
        ]])
        return self.predict(features)[0]