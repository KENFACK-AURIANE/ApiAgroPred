# app.py corrigé

import pandas as pd
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
# from sklearn.preprocessing import OneHotEncoder # Plus besoin d'importer la classe ici, on utilise l'instance chargée

# --- 1. Initialisation de l'application Flask ---
app = Flask(__name__)
CORS(app)

# Variables globales pour le modèle et l'encodeur chargés
model = None
encoder = None

# --- 2. Chargement du Modèle et de l'Encodeur ---
MODEL_PATH = "xgboost_model.pkl"
ENCODER_PATH = "oneHotEncoder.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
        print("Modèle et encodeur chargés avec succès.")
    except Exception as e:
        print(f"Erreur lors du chargement des fichiers Joblib: {e}")
        exit()
else:
    print(f"Erreur: Fichiers modèle ({MODEL_PATH} ou {ENCODER_PATH}) non trouvés.")
    exit()

# --- 3. Fonction de Prétraitement (Corrigée) ---
def preprocess_new_data(data_dict):
    """
    Prétraite un dictionnaire de nouvelles données brutes pour le modèle.
    Utilise l'encodeur global 'encoder' déjà entraîné.
    """
    df_new = pd.DataFrame([data_dict])
    
    # Définition des colonnes (doit correspondre à l'entraînement d'origine)
    bools_cols = ['Fertilizer_Used', 'Irrigation_Used']
    categorical_cols = ['Region', 'Soil_Type','Crop', 'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition']
    
    # 2a. Convertir les booléens/int si nécessaire (s'assurer qu'ils sont traités comme des catégories si c'est le cas)
    for col in bools_cols:
        if col in df_new.columns:
            df_new[col] = df_new[col].astype(int) # Important : traiter comme chaîne pour l'encodage OHE

    # 2b. S'assurer que les autres colonnes catégorielles sont des chaînes
    for col in ['Region', 'Soil_Type','Crop', 'Weather_Condition']:
         if col in df_new.columns:
            df_new[col] = df_new[col].astype(str)

    # Filtrer les colonnes qui existent réellement dans l'input
    cols_to_encode = [col for col in categorical_cols if col in df_new.columns]
    
    # --- Utilisation de l'encodeur global (CORRECTION CLÉ) ---
    # On utilise TRANSFORM, pas fit_transform.
    # Et on utilise l'instance 'encoder' chargée depuis le fichier .pkl
    encoded_data = encoder.transform(df_new[cols_to_encode])

    # Création d'un DataFrame encodé
    encoded_df_new = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cols_to_encode), index=df_new.index)

    # Identifier les colonnes numériques qui ne sont PAS dans les catégories
    numeric_cols = [col for col in df_new.columns if col not in categorical_cols]
    
    # Fusion avec le DataFrame d'origine
    # On concatène les colonnes numériques et les colonnes encodées
    final_X = pd.concat([df_new[numeric_cols], encoded_df_new], axis=1)

    return final_X

# --- 4. Endpoint de Prédiction (L'URL de l'API) ---
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # S'assurer que data est un dictionnaire
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid input format, expected a JSON object/dictionary."}), 400
            
        processed_data = preprocess_new_data(data)
        prediction = model.predict(processed_data)[0]
        return jsonify({"prediction_yield_tons_per_hectare": float(prediction)})
    except Exception as e:
        # Capture l'exception exacte et la renvoie pour un meilleur débogage sur Render
        app.logger.error("Erreur lors de la prédiction : %s", e)
        return jsonify({"error": "Internal server error during prediction", "details": str(e)}), 500


# --- 5. Lancement de l'application (Configuration pour Render) ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)



# import pandas as pd
# import joblib
# from flask import Flask, request, jsonify
# from flask_cors import CORS # Importé pour permettre l'accès depuis un site web
# import os
# from sklearn.preprocessing import OneHotEncoder
# # j'ai modifie App.py en app.py
# # --- 1. Initialisation de l'application Flask ---
# app = Flask(__name__)
# CORS(app) # Active le partage de ressources entre origines différentes (essentiel pour un site web externe)

# # --- 2. Chargement du Modèle et de l'Encodeur ---
# # Définit les noms des fichiers que nous allons charger
# MODEL_PATH = "xgboost_model.pkl"
# ENCODER_PATH = "oneHotEncoder.pkl"

# # Tente de charger les fichiers au démarrage de l'API
# if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
#     model = joblib.load(MODEL_PATH)
#     encoder = joblib.load(ENCODER_PATH) 
#     print("Modèle et encodeur chargés avec succès.")
# else:
#     print(f"Erreur: Fichiers modèle ({MODEL_PATH} ou {ENCODER_PATH}) non trouvés.")
#     # Si les fichiers essentiels sont manquants, l'API ne peut pas fonctionner. On arrête l'exécution.
#     exit()

# # --- 3. Fonction de Prétraitement ---
# def preprocess_new_data(data_dict):
#     """
#     Prétraite un dictionnaire de nouvelles données brutes pour le modèle.
#     """
#     # Convertit le dictionnaire JSON entrant en DataFrame pandas
#     df_new = pd.DataFrame([data_dict])
    
#     # Définit les colonnes qui doivent être encodées (doit correspondre à l'entraînement)
#     # 2a. Convertir les booléens en int (comme dans le script d'entrainement)
#     bools_cols = ['Fertilizer_Used', 'Irrigation_Used']
#     for col in bools_cols:
#         if col in df_new.columns:
#             df_new[col] = df_new[col].astype(int)
#     # 2b. Colonnes catégorielles et numériques
#     categorical_cols = ['Region', 'Soil_Type','Crop', 'Fertilizer_Used', 'Irrigation_Used', 'Weather_Condition']
#     # On s'assure que toutes ces colonnes existent bien
#     categorical_cols = [col for col in categorical_cols if col in df_new.columns]

#     # On remplace les valeurs manquantes par 'Unknown' pour éviter les erreurs d'encodage
#     #df_new[categorical_cols] = df_new[categorical_cols].fillna('Unknown')

#     # Encodage one-hot
#     encoder = OneHotEncoder(sparse_output=False)
#     encoded_data = encoder.fit_transform(df_new[categorical_cols])

#     # Création d'un DataFrame encodé
#     encoded_df_new = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_cols), index=df_new.index)

#     # Fusion avec le DataFrame d'origine
#     df_new = pd.concat([df_new.drop(categorical_cols, axis=1), encoded_df_new], axis=1)

#     # Combine les colonnes numériques et encodées pour former l'input final pour le modèle
#     # reset_index(drop=True) est utilisé pour s'assurer que les index correspondent avant la concaténation
#     final_X = pd.concat([df_new.drop(categorical_cols).reset_index(drop=True), encoded_df_new], axis=1)

#     return final_X

# # --- 4. Endpoint de Prédiction (L'URL de l'API) ---
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Récupère les données envoyées par l'utilisateur via une requête POST (le JSON)
#     data = request.get_json(force=True)
    
#     # Utilise notre fonction de prétraitement pour préparer les données
#     processed_data = preprocess_new_data(data)
    
#     # Fait la prédiction en utilisant le modèle chargé
#     prediction = model.predict(processed_data)[0] # [0] pour extraire la valeur unique

#     # Formate le résultat de la prédiction dans un objet JSON standard
#     return jsonify({"prediction_yield_tons_per_hectare": float(prediction)})

# # --- 5. Lancement de l'application (Configuration pour Render) ---
# if __name__ == '__main__':
#     # Render utilise une variable d'environnement 'PORT' pour dire à l'API quel port utiliser.
#     # On utilise os.environ.get("PORT", 5000) pour utiliser la variable de Render, 
#     # ou 5000 si on est sur notre machine locale.
#     port = int(os.environ.get("PORT", 5000))
    
#     # 'host="0.0.0.0"' rend l'API accessible publiquement sur le serveur Render.
#     app.run(host='0.0.0.0', port=port, debug=False)
