import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from database import DatabaseManager
from ml_model import HousingPredictor

# Configuration MLflow
mlflow.set_tracking_uri("http://http://20.36.136.43:5000")

st.set_page_config(
    page_title="Prédiction Prix Immobilier",
    page_icon="🏠",
    layout="wide"
)

def main():
    st.title("🏠 Système de Prédiction Prix Immobilier")
    st.sidebar.title("Navigation")
    
    # Initialisation de la base de données
    db = DatabaseManager()
    
    # Menu de navigation
    page = st.sidebar.selectbox(
        "Choisir une page",
        ["Prédiction", "Entraînement Modèle", "Données", "Historique MLflow"]
    )
    
    if page == "Prédiction":
        show_prediction_page(db)
    elif page == "Entraînement Modèle":
        show_training_page(db)
    elif page == "Données":
        show_data_page(db)
    elif page == "Historique MLflow":
        show_mlflow_page()

def show_prediction_page(db):
    st.header("Prédiction de Prix")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Caractéristiques du bien")
        surface = st.number_input("Surface (m²)", min_value=20, max_value=500, value=100)
        rooms = st.number_input("Nombre de pièces", min_value=1, max_value=10, value=3)
        bedrooms = st.number_input("Nombre de chambres", min_value=1, max_value=8, value=2)
        age = st.number_input("Âge du bien (années)", min_value=0, max_value=100, value=10)
        location_score = st.slider("Score localisation (1-10)", 1, 10, 7)
        
    with col2:
        st.subheader("Paramètres avancés")
        has_garden = st.checkbox("Jardin")
        has_parking = st.checkbox("Parking")
        has_balcony = st.checkbox("Balcon")
        energy_class = st.selectbox("Classe énergétique", ["A", "B", "C", "D", "E", "F", "G"])
        
    if st.button("Prédire le prix", type="primary"):
        # Chargement du modèle
        try:
            predictor = HousingPredictor()
            
            # Préparation des données
            features = np.array([[
                surface, rooms, bedrooms, age, location_score,
                int(has_garden), int(has_parking), int(has_balcony),
                ord(energy_class) - ord('A')
            ]])
            
            # Prédiction
            prediction = predictor.predict(features)[0]
            
            st.success(f"Prix estimé: {prediction:,.0f} €")
            
            # Sauvegarde en base
            db.save_prediction({
                'surface': surface,
                'rooms': rooms,
                'bedrooms': bedrooms,
                'age': age,
                'location_score': location_score,
                'has_garden': has_garden,
                'has_parking': has_parking,
                'has_balcony': has_balcony,
                'energy_class': energy_class,
                'predicted_price': prediction
            })
            
        except Exception as e:
            st.error(f"Erreur lors de la prédiction: {str(e)}")

def show_training_page(db):
    st.header("Entraînement du Modèle")
    
    if st.button("Générer données d'exemple"):
        generate_sample_data(db)
        st.success("Données générées avec succès!")
    
    if st.button("Entraîner le modèle", type="primary"):
        with st.spinner("Entraînement en cours..."):
            train_model(db)

def generate_sample_data(db):
    np.random.seed(42)
    n_samples = 1000
    
    data = []
    for i in range(n_samples):
        surface = np.random.normal(120, 40)
        surface = max(30, min(300, surface))
        
        rooms = np.random.poisson(4) + 1
        bedrooms = min(rooms - 1, np.random.poisson(2) + 1)
        age = np.random.exponential(15)
        location_score = np.random.normal(6, 2)
        location_score = max(1, min(10, location_score))
        
        has_garden = np.random.choice([0, 1], p=[0.6, 0.4])
        has_parking = np.random.choice([0, 1], p=[0.4, 0.6])
        has_balcony = np.random.choice([0, 1], p=[0.5, 0.5])
        energy_class = np.random.choice(list(range(7)), p=[0.05, 0.1, 0.15, 0.3, 0.25, 0.1, 0.05])
        
        # Prix basé sur les caractéristiques
        price = (surface * 3000 + 
                rooms * 15000 + 
                bedrooms * 10000 - 
                age * 1000 + 
                location_score * 20000 +
                has_garden * 25000 +
                has_parking * 15000 +
                has_balcony * 8000 -
                energy_class * 5000 +
                np.random.normal(0, 20000))
        
        price = max(50000, price)
        
        data.append({
            'surface': surface,
            'rooms': rooms,
            'bedrooms': bedrooms,
            'age': age,
            'location_score': location_score,
            'has_garden': has_garden,
            'has_parking': has_parking,
            'has_balcony': has_balcony,
            'energy_class': energy_class,
            'price': price
        })
    
    db.save_training_data(data)

def train_model(db):
    # Récupération des données
    data = db.get_training_data()
    
    if not data:
        st.error("Aucune donnée d'entraînement disponible")
        return
    
    df = pd.DataFrame(data)
    
    # Préparation des données
    X = df[['surface', 'rooms', 'bedrooms', 'age', 'location_score', 
           'has_garden', 'has_parking', 'has_balcony', 'energy_class']]
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # MLflow experiment
    experiment_name = "housing_price_prediction"
    try:
        mlflow.create_experiment(experiment_name)
    except:
        pass
    
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        # Entraînement
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Métriques
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Logging MLflow
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")
        
        # Sauvegarde locale
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/housing_model.pkl")
        
        # Affichage des résultats
        col1, col2 = st.columns(2)
        with col1:
            st.metric("MSE", f"{mse:,.0f}")
        with col2:
            st.metric("R²", f"{r2:.3f}")
        
        # Graphique
        fig = px.scatter(x=y_test, y=y_pred, 
                        labels={'x': 'Prix réel', 'y': 'Prix prédit'},
                        title="Prédiction vs Réalité")
        fig.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), 
                     x1=y_test.max(), y1=y_test.max())
        st.plotly_chart(fig)

def show_data_page(db):
    st.header("Données")
    
    tab1, tab2 = st.tabs(["Données d'entraînement", "Prédictions"])
    
    with tab1:
        training_data = db.get_training_data()
        if training_data:
            df = pd.DataFrame(training_data)
            st.dataframe(df)
            
            # Statistiques
            st.subheader("Statistiques")
            st.write(df.describe())
        else:
            st.info("Aucune donnée d'entraînement")
    
    with tab2:
        predictions = db.get_predictions()
        if predictions:
            df_pred = pd.DataFrame(predictions)
            st.dataframe(df_pred)
        else:
            st.info("Aucune prédiction")

def show_mlflow_page():
    st.header("MLflow Tracking")
    st.info("Interface MLflow disponible à: http://http://20.36.136.43:5000")
    
    # Tentative d'affichage des runs
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        
        for exp in experiments:
            st.subheader(f"Expérience: {exp.name}")
            runs = client.search_runs(exp.experiment_id)
            
            if runs:
                runs_data = []
                for run in runs:
                    runs_data.append({
                        'Run ID': run.info.run_id[:8],
                        'Status': run.info.status,
                        'MSE': run.data.metrics.get('mse', 'N/A'),
                        'R²': run.data.metrics.get('r2', 'N/A'),
                        'Date': run.info.start_time
                    })
                
                st.dataframe(pd.DataFrame(runs_data))
    except Exception as e:
        st.error(f"Erreur connexion MLflow: {str(e)}")

if __name__ == "__main__":
    main()