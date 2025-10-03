import joblib
import numpy as np

import pandas as pd
from sklearn.preprocessing import StandardScaler


model = joblib.load("model/final_model.joblib")

# Exemple d'entrée utilisateur
print("Entrez les informations suivantes pour estimer les charges :")
age = float(input("Âge : "))
bmi = float(input("IMC (BMI) : "))
children = int(input("Nombre d'enfants : "))
sex = input("Sexe (male/female) : ").lower()
smoker = input("Fumeur ? (yes/no) : ").lower()
region = input("Région (northeast/northwest/southeast/southwest) : ").lower()



num_cols = ["age", "bmi", "children"]

# Créer DataFrame pour la prédiction
data = pd.DataFrame({
    "age": [age],
    "bmi": [bmi],
    "children": [children],
    "sex_female": [1 if sex=="female" else 0],
    "sex_male": [1 if sex=="male" else 0],
    "smoker_no": [1 if smoker=="no" else 0],
    "smoker_yes": [1 if smoker=="yes" else 0],
    "region_northeast": [1 if region=="northeast" else 0],
    "region_northwest": [1 if region=="northwest" else 0],
    "region_southeast": [1 if region=="southeast" else 0],
    "region_southwest": [1 if region=="southwest" else 0],
})

# Prédiction log charges -> revenir à l'échelle originale
log_pred = model.predict(data)
pred_charges = np.exp(log_pred)

print(f"\nEstimation des charges d'assurance : {pred_charges[0]:,.2f} €")

