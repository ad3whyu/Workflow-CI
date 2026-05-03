import argparse
import pandas as pd
import numpy as np
import os
import dagshub
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def train_model(n_estimators, max_depth, min_samples_split):
    # 1. Inisialisasi DagsHub
    dagshub.init(repo_owner='ad3whyu', repo_name='Eksperimen_SML_Ade-Wahyu-Warpudin', mlflow=True)
    mlflow.set_experiment("Car_Price_CI_Retraining")

    # 2. Load Data (Pastikan folder dataset ada di dalam folder MLProject ini)
    # Path disesuaikan untuk dijalankan di dalam folder MLProject
    path = "used_car_price_dataset_extended_preprocessing/"
    train_df = pd.read_csv(os.path.join(path, "data_train_ready.csv"))
    test_df = pd.read_csv(os.path.join(path, "data_test_ready.csv"))

    X_train = train_df.drop(columns=['price_usd'])
    y_train = train_df['price_usd']
    X_test = test_df.drop(columns=['price_usd'])
    y_test = test_df['price_usd']

    with mlflow.start_run(run_name="CI_Retraining_Run"):
        # 3. Model Training dengan parameter dari MLProject
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)

        # 4. Evaluasi
        y_pred = model.fit(X_train, y_train).predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        # 5. Manual Logging
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("R2_Score", r2)
        mlflow.log_metric("MAE", mae)

        # 6. Log Artifact (Contoh: Plot Feature Importance)
        plt.figure(figsize=(10, 6))
        feat_importances = pd.Series(model.feature_importances_, index=X_train.columns)
        feat_importances.nlargest(10).plot(kind='barh')
        plt.tight_layout()
        plt.savefig("feature_importance_ci.png")
        mlflow.log_artifact("feature_importance_ci.png")

        # 7. Log Model
        mlflow.sklearn.log_model(model, "model_retrained")

        print(f"Retraining Selesai! R2 Score: {r2:.4f}")

if __name__ == "__main__":
    # Mengambil argumen dari MLProject
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    args = parser.parse_args()

    train_model(args.n_estimators, args.max_depth, args.min_samples_split)