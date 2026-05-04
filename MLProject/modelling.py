import os
import shutil
import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor

def jalankan_retraining(n_estimators, max_depth, min_samples_split):
    # Kunci URI ke DagsHub (sudah diamankan via environment variables GitHub Actions)
    mlflow.set_tracking_uri("https://dagshub.com/ad3whyu/Eksperimen_SML_Ade-Wahyu-Warpudin.mlflow")

    # Bikin path data otomatis
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "used_car_price_dataset_extended_preprocessing", "data_train_ready.csv")
    
    # Cek ketersediaan dataset
    if not os.path.exists(data_path):
         # Fallback 1: Jika dijalankan di GitHub Actions
         data_path = "used_car_price_dataset_extended_preprocessing/data_train_ready.csv"
         
    train_df = pd.read_csv(data_path)
    X_train = train_df.drop(columns=['price_usd'])
    y_train = train_df['price_usd']

    print(f"Memulai training: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}")
    
    rf_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    
    mlflow.sklearn.autolog(log_models=False)
    
    # ─── PERBAIKAN KRUSIAL: JANGAN START RUN BARU ───
    # Karena mlflow run sudah membukakan pintu untuk kita, kita cukup mengambil Run ID yang sedang aktif
    active_run = mlflow.active_run()
    
    # Jika dijalankan via mlflow run, active_run pasti ada nilainya
    if active_run:
        print(f"Menggunakan Run ID yang sudah dibuat mlflow: {active_run.info.run_id}")
        rf_model.fit(X_train, y_train)
        
        mlflow.sklearn.log_model(rf_model, "model_artifacts")
        
        lokasi_aman = os.path.expanduser("~/model_retrained")
        if os.path.exists(lokasi_aman):
            shutil.rmtree(lokasi_aman)
            
        mlflow.sklearn.save_model(rf_model, lokasi_aman)
        print(f"✅ Folder model fisik berhasil diselamatkan ke: {lokasi_aman}")
        
    else:
        # Fallback jika dijalankan manual (python modelling.py)
        with mlflow.start_run(run_name="Automated_Retraining"):
            rf_model.fit(X_train, y_train)
            
            mlflow.sklearn.log_model(rf_model, "model_artifacts")
            
            lokasi_aman = os.path.expanduser("~/model_retrained")
            if os.path.exists(lokasi_aman):
                shutil.rmtree(lokasi_aman)
                
            mlflow.sklearn.save_model(rf_model, lokasi_aman)
            print(f"✅ Folder model fisik berhasil diselamatkan ke: {lokasi_aman}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--min_samples_split", type=int, default=2)
    args = parser.parse_args()

    jalankan_retraining(args.n_estimators, args.max_depth, args.min_samples_split)