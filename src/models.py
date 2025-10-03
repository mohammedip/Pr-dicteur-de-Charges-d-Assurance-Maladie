from preprocess import X_train, X_test, y_train, y_test ,pd
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np



def model_performance(model , X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    y_test2 =np.exp(y_test)
    y_pred2 =np.exp(y_pred)

    rmse = np.sqrt(mean_squared_error(y_test2, y_pred2))
    mae = mean_absolute_error(y_test2, y_pred2)
    r2 = r2_score(y_test, y_pred)

    return {"Model": name, "RMSE": rmse, "MAE": mae, "R²": r2}


results =[]

#training models 
lr = LinearRegression()
results.append(model_performance(lr, X_train, y_train, X_test, y_test, "Linear Regression"))

rf = RandomForestRegressor(random_state=42)
results.append(model_performance(rf, X_train, y_train, X_test, y_test, "Random Forest"))

xgb = XGBRegressor(random_state=42)
results.append(model_performance(xgb, X_train, y_train, X_test, y_test, "XGB"))

svr = SVR(kernel='rbf', C=100, epsilon=0.1)
results.append(model_performance(svr, X_train, y_train, X_test, y_test, "SVR"))



results.append({"Model": "--" , "RMSE": "--", "MAE": "--", "R²": "--"})



#use gridsearch or RandomizedSearchCV then train models again

# Hyperparameters RF
param_rf = {
    "n_estimators": [100, 200, 500],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10]
}

grid_rf = GridSearchCV(rf, param_rf, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
grid_rf.fit(X_train, y_train)
rf_best = grid_rf.best_estimator_

results.append(model_performance(rf_best, X_train, y_train, X_test, y_test, "Random Forest (optimized)"))


# Hyperparameters XGB
param_xgb = {
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "n_estimators": [100, 200, 500],
    "max_depth": [3, 5, 7, 10],
    "subsample": [0.6, 0.8, 1.0],
}

rand_xgb = GridSearchCV(
    xgb, param_xgb, cv=5, scoring="neg_mean_squared_error",
    n_jobs=15, verbose=1
)
rand_xgb.fit(X_train, y_train)
xgb_best = rand_xgb.best_estimator_

results.append(model_performance(xgb_best, X_train, y_train, X_test, y_test, "XGB (optimized)"))

# Hyperparameters SVR
param_svr = {
    "C": [0.1, 1, 10, 100],
    "epsilon": [0.01, 0.1, 0.5],
    "kernel": ["linear", "rbf", "poly"]
}

grid_svr = GridSearchCV(svr, param_svr, cv=5, scoring="neg_mean_squared_error", n_jobs=-1, verbose=1)
grid_svr.fit(X_train, y_train)
svr_best = grid_svr.best_estimator_

results.append(model_performance(svr_best, X_train, y_train, X_test, y_test, "SVR (optimized)"))


