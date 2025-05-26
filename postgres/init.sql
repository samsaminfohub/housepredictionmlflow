-- Création de la base de données
CREATE DATABASE housing_db;

\\c housing_db;

-- Table des données d'entraînement
CREATE TABLE training_data (
    id SERIAL PRIMARY KEY,
    surface FLOAT NOT NULL,
    rooms INTEGER NOT NULL,
    bedrooms INTEGER NOT NULL,
    age FLOAT NOT NULL,
    location_score FLOAT NOT NULL,
    has_garden BOOLEAN NOT NULL,
    has_parking BOOLEAN NOT NULL,
    has_balcony BOOLEAN NOT NULL,
    energy_class INTEGER NOT NULL,
    price FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Table des prédictions
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    surface FLOAT NOT NULL,
    rooms INTEGER NOT NULL,
    bedrooms INTEGER NOT NULL,
    age FLOAT NOT NULL,
    location_score FLOAT NOT NULL,
    has_garden BOOLEAN NOT NULL,
    has_parking BOOLEAN NOT NULL,
    has_balcony BOOLEAN NOT NULL,
    energy_class VARCHAR(1) NOT NULL,
    predicted_price FLOAT NOT NULL,
    prediction_date TIMESTAMP DEFAULT NOW()
);

-- Index pour optimiser les requêtes
CREATE INDEX idx_predictions_date ON predictions(prediction_date);
CREATE INDEX idx_training_data_price ON training_data(price);