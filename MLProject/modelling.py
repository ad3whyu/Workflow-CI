import os
import shutil
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor

def jalankan_retraining(n_estimators, max_depth, min_samples_split):
    # --- TAMBAHKAN KUNCI DAGSHUB ---
    # Agar riwayat GitHub Actions juga tercatat secara online
    mlflow.set_tracking_uri("https://dagshub.com/ad3whyu/Eksperimen_SML_Ade-Wahyu-Warpudin.mlflow")
    mlflow.set_experiment("Car_Price_Retraining_CICD")

    # 1. LOAD DATASET
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
    
    # Matikan log_models bawaan agar tidak bentrok dengan save_model kita
    mlflow.sklearn.autolog(log_models=False)
    
    with mlflow.start_run(run_name="Automated_Retraining"):
        rf_model.fit(X_train, y_train)
        
        # --- PERBAIKAN UTAMA: LOG KE DAGSHUB & SIMPAN FISIK LOKAL ---
        
        # A. Upload ke DagsHub sebagai rekam jejak
        mlflow.sklearn.log_model(rf_model, "model_artifacts")
        
        # B. Bikin folder fisik 'model_retrained' agar Docker bisa melakukan build!
        if os.path.exists("model_retrained"):
            shutil.rmtree("model_retrained") # Hapus folder lama biar tidak error tertimpa
            
        mlflow.sklearn.save_model(rf_model, "model_retrained")
        
        print("✅ Retraining selesai! Folder fisik 'model_retrained' berhasil dibuat untuk Docker.")

if __name__ == "__main__":
    # 3. TANGKAP PARAMETER DARI TERMINAL / MLPROJECT
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    args = parser.parse_args()

    jalankan_retraining(args.n_estimators, args.max_depth, args.min_samples_split)