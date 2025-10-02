from preprocess import X_train, X_test, y_train, y_test 
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np



lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

y_test2 =np.exp(y_test)
y_pred2 =np.exp(y_pred)

lr_rmse = np.sqrt(mean_squared_error(y_test2, y_pred2))
lr_mae = mean_absolute_error(y_test2, y_pred2)
lr_r2 = r2_score(y_test, y_pred)


rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test2, y_pred2))
rf_mae = mean_absolute_error(y_test2, y_pred2)
rf_r2 = r2_score(y_test, y_pred)


xgb = XGBRegressor(random_state=42)
xgb.fit(X_train, y_train)

y_pred = xgb.predict(X_test)

xgb_rmse = np.sqrt(mean_squared_error(y_test2, y_pred2))
xgb_mae = mean_absolute_error(y_test2, y_pred2)
xgb_r2 = r2_score(y_test, y_pred)


svr_model = SVR(kernel='rbf', C=100, epsilon=0.1)
svr_model.fit(X_train, y_train)

y_pred = svr_model.predict(X_test)

svr_rmse = np.sqrt(mean_squared_error(y_test2, y_pred2))
svr_mae = mean_absolute_error(y_test2, y_pred2)
svr_r2 = r2_score(y_test, y_pred)


print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
print("-" * 50)

print(f"{'Linear Regression':<20} {lr_rmse:<10.4f} {lr_mae:<10.4f} {lr_r2:<10.4f}")
print(f"{'Random Forest':<20} {rf_rmse:<10.4f} {rf_mae:<10.4f} {rf_r2:<10.4f}")
print(f"{'XGBoost':<20} {xgb_rmse:<10.4f} {xgb_mae:<10.4f} {xgb_r2:<10.4f}")
print(f"{'SVR':<20} {svr_rmse:<10.4f} {svr_mae:<10.4f} {svr_r2:<10.4f}")


