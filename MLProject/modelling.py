import os
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor

def jalankan_retraining(n_estimators, max_depth, min_samples_split):
    # 1. LOAD DATASET
    # Pastikan path ini sesuai dengan struktur folder kamu di GitHub
    data_dir = "used_car_price_dataset_extended_preprocessing/"
    train_df = pd.read_csv(os.path.join(data_dir, "data_train_ready.csv"))
    
    X_train = train_df.drop(columns=['price_usd'])
    y_train = train_df['price_usd']

    # 2. TRAINING DENGAN PARAMETER DARI MLFLOW RUN
    print(f"Memulai training dengan n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}")
    
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="Automated_Retraining"):
        rf_model.fit(X_train, y_train)
        
        # Simpan model dengan nama folder "model_retrained" sesuai kebutuhan Docker kamu
        mlflow.sklearn.log_model(rf_model, "model_retrained")
        print("Retraining selesai dan model disimpan di folder model_retrained.")

if __name__ == "__main__":
    # 3. TANGKAP PARAMETER DARI TERMINAL / MLPROJECT
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    args = parser.parse_args()

    jalankan_retraining(args.n_estimators, args.max_depth, args.min_samples_split)