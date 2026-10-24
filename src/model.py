import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def train_and_evaluate(X, y):
    # Split into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 1. Baseline Model: Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_r2 = r2_score(y_test, lr_preds)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_preds))

    print(
        f"Baseline Model (Linear Regression) -> R^2 on test set: {lr_r2:.2f} | RMSE: {lr_rmse:.2f}%"
    )

    # 2. Main Model: Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
    rf_mae = mean_absolute_error(y_test, rf_preds)

    print(
        f"Final Model (Random Forest)        -> R^2 on test set: {rf_r2:.2f} | RMSE: {rf_rmse:.2f}% | MAE: {rf_mae:.2f}%"
    )

    return rf_model, X.columns, y_test, rf_preds
