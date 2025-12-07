import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS # Importé pour permettre l'accès depuis un site web
import os
# j'ai modifie App.py en app.py
# --- 1. Initialisation de l'application Flask ---
app = Flask(__name__)
CORS(app) # Active le partage de ressources entre origines différentes (essentiel pour un site web externe)

# --- 2. Chargement du Modèle et de l'Encodeur ---
# Définit les noms des fichiers que nous allons charger
MODEL_PATH = "xgboost_model.pkl"
ENCODER_PATH = "oneHotEncoder.pkl"

# Tente de charger les fichiers au démarrage de l'API
if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    model = joblib.load(MODEL_PATH)
    encoder = joblib.load(ENCODER_PATH) 
    print("Modèle et encodeur chargés avec succès.")
else:
    print(f"Erreur: Fichiers modèle ({MODEL_PATH} ou {ENCODER_PATH}) non trouvés.")
    # Si les fichiers essentiels sont manquants, l'API ne peut pas fonctionner. On arrête l'exécution.
    exit()

# --- 3. Fonction de Prétraitement ---
def preprocess_new_data(data_dict):
    """
    Prétraite un dictionnaire de nouvelles données brutes pour le modèle.
    """
    # Convertit le dictionnaire JSON entrant en DataFrame pandas
    df_new = pd.DataFrame([data_dict])
    
    # Définit les colonnes qui doivent être encodées (doit correspondre à l'entraînement)
    # 2a. Convertir les booléens en int (comme dans le script d'entrainement)
    bools_cols = ['Fertilizer_Used', 'Irrigation_Used']
    for col in bools_cols:
        if col in df_new.columns:
            df_new[col] = df_new[col].astype(int)
    # 2b. Colonnes catégorielles et numériques
    categorical_cols = ['Region', 'Soil_Type', 'Crop',   'Weather_Condition']
    # numerical_cols = ['Rainfall_mm', 'Temperature_Celsius', 'Days_to_Harvest']

    # Applique l'encodeur OneHotEncoder sauvegardé
    # L'utilisation de .transform() garantit que l'ordre des colonnes et les catégories sont les mêmes que lors du .fit()
    encoded_data = encoder.transform(df_new[categorical_cols])

    # Crée un DataFrame à partir des données encodées avec les bons noms de colonnes
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols))

    # Combine les colonnes numériques et encodées pour former l'input final pour le modèle
    # reset_index(drop=True) est utilisé pour s'assurer que les index correspondent avant la concaténation
    final_X = pd.concat([df_new.drop(categorical_cols).reset_index(drop=True), encoded_df], axis=1)

    return final_X

# --- 4. Endpoint de Prédiction (L'URL de l'API) ---
@app.route('/predict', methods=['POST'])
def predict():
    # Récupère les données envoyées par l'utilisateur via une requête POST (le JSON)
    data = request.get_json(force=True)
    
    # Utilise notre fonction de prétraitement pour préparer les données
    processed_data = preprocess_new_data(data)
    
    # Fait la prédiction en utilisant le modèle chargé
    prediction = model.predict(processed_data)[0] # [0] pour extraire la valeur unique

    # Formate le résultat de la prédiction dans un objet JSON standard
    return jsonify({"prediction_yield_tons_per_hectare": float(prediction)})

# --- 5. Lancement de l'application (Configuration pour Render) ---
if __name__ == '__main__':
    # Render utilise une variable d'environnement 'PORT' pour dire à l'API quel port utiliser.
    # On utilise os.environ.get("PORT", 5000) pour utiliser la variable de Render, 
    # ou 5000 si on est sur notre machine locale.
    port = int(os.environ.get("PORT", 5000))
    
    # 'host="0.0.0.0"' rend l'API accessible publiquement sur le serveur Render.
    app.run(host='0.0.0.0', port=port, debug=False)
