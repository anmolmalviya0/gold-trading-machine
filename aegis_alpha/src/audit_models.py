import os
import joblib
import numpy as np

MODELS_DIR = "/Users/anmol/Desktop/gold/aegis_alpha/models"

def audit_models():
    print("🔍 GLOBAL MODEL FEATURE AUDIT")
    print("==============================")
    
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith(".pkl")]
    for f in sorted(files):
        path = os.path.join(MODELS_DIR, f)
        try:
            model = joblib.load(path)
            # LightGBM model typically has .n_features_ attribute
            if hasattr(model, 'n_features_'):
                n_feat = model.n_features_
                status = "✅ 15" if n_feat == 15 else f"❌ {n_feat}"
                print(f"{f:<30} : {status}")
            else:
                print(f"{f:<30} : ❓ NO FEATURE ATTR")
        except Exception as e:
            print(f"{f:<30} : 💥 FAILED TO LOAD: {e}")

if __name__ == "__main__":
    audit_models()
