import matplotlib.pyplot as plt
import numpy as np
from models import X_test, y_test ,pd ,rf_best ,xgb_best ,svr_best ,results
import joblib



# Prédictions des modèles optimisés
y_pred_rf = rf_best.predict(X_test)
y_pred_xgb = xgb_best.predict(X_test)
y_pred_svr = svr_best.predict(X_test)

# Pour visualiser sur l’échelle originale
y_test_orig = np.exp(y_test)
y_pred_rf_orig = np.exp(y_pred_rf)
y_pred_xgb_orig = np.exp(y_pred_xgb)
y_pred_svr_orig = np.exp(y_pred_svr)


#graphiques de résidus avec Matplotlib
def plot_residuals(y_true, y_pred, model_name):
    residuals = y_true - y_pred
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Prédictions")
    plt.ylabel("Résidus (y_true - y_pred)")
    plt.title(f"Graphique des résidus - {model_name}")
    plt.show()

plot_residuals(y_test_orig, y_pred_rf_orig, "Random Forest (optimized)")

plot_residuals(y_test_orig, y_pred_xgb_orig, "XGBoost (optimized)")

# SVR
plot_residuals(y_test_orig, y_pred_svr_orig, "SVR (optimized)")



def plot_pred_vs_true(y_true, y_pred, model_name):
    plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel("Valeurs réelles")
    plt.ylabel("Valeurs prédites")
    plt.title(f"Prédictions vs Réelles - {model_name}")
    plt.show()

# Random Forest
plot_pred_vs_true(y_test_orig, y_pred_rf_orig, "Random Forest (optimized)")

# XGBoost
plot_pred_vs_true(y_test_orig, y_pred_xgb_orig, "XGBoost (optimized)")

# SVR
plot_pred_vs_true(y_test_orig, y_pred_svr_orig, "SVR (optimized)")

# #tableau des resultats
df_results = pd.DataFrame(results)
print(df_results)

joblib.dump(rf_best, "model/final_model.joblib")
