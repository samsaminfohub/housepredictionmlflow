# Projet MLOps - Prédiction Prix Immobilier

## Description
Application complète de machine learning pour la prédiction de prix immobilier avec:
- Interface Streamlit
- Base de données PostgreSQL
- Tracking MLflow
- Déploiement Docker

## Architecture
- **Streamlit**: Interface utilisateur et API
- **PostgreSQL**: Stockage des données
- **MLflow**: Tracking des expériences ML
- **Docker**: Conteneurisation et orchestration

## Installation et Démarrage

### Prérequis
- Docker
- Docker Compose

### Lancement
```bash
# Cloner le projet
git clone <repository>
cd ml_housing_project

# Démarrer tous les services
docker-compose up -d

# Voir les logs
docker-compose logs -f
```

### Accès aux services
- **Streamlit**: http://localhost:8501
- **MLflow**: http://localhost:5000
- **PostgreSQL**: localhost:5432

## Utilisation

1. **Génération de données**: Aller dans "Entraînement Modèle" et cliquer sur "Générer données d'exemple"
2. **Entraînement**: Cliquer sur "Entraîner le modèle"
3. **Prédiction**: Utiliser l'onglet "Prédiction" pour estimer des prix
4. **Suivi**: Consulter MLflow pour le tracking des expériences

## Fonctionnalités

### Interface Streamlit
- Prédiction interactive de prix
- Entraînement de modèles
- Visualisation des données
- Interface MLflow intégrée

### Base de données
- Stockage des données d'entraînement
- Historique des prédictions
- Optimisation avec index

### MLflow
- Tracking des expériences
- Comparaison des modèles
- Stockage des artefacts

## Structure du projet
```
ml_housing_project/
├── app/                    # Code Streamlit
├── mlflow/                 # Configuration MLflow
├── postgres/               # Scripts SQL
├── models/                 # Modèles sauvegardés
├── docker-compose.yml      # Orchestration
└── README.md
```

## Commandes utiles

```bash
# Arrêter les services
docker-compose down

# Reconstruire les images
docker-compose build

# Voir les logs d'un service
docker-compose logs streamlit

# Nettoyer les volumes
docker-compose down -v
```

## Développement

Pour développer localement:
1. Installer les dépendances: `pip install -r app/requirements.txt`
2. Démarrer PostgreSQL et MLflow avec Docker
3. Lancer Streamlit: `streamlit run app/streamlit_app.py`

## Notes techniques
- Le modèle utilise RandomForest pour la régression
- Les données sont générées synthétiquement si aucune donnée réelle n'est disponible
- L'application supporte le hot-reload en développement