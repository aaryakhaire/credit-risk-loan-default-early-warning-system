import lightgbm as lgb
import joblib
import pandas as pd
import numpy as np
import shap

MODEL_PATH = "models/credit_risk_model.txt"
METADATA_PATH = "models/model_metadata.pkl"

model = lgb.Booster(model_file=MODEL_PATH)
metadata = joblib.load(METADATA_PATH)

feature_cols = metadata['feature_cols']
cat_features = metadata['cat_features']
optimal_threshold = metadata['optimal_threshold']

explainer = shap.TreeExplainer(model)

def predict_risk(applicant_dict: dict):
    df = pd.DataFrame([applicant_dict])
    
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    
    df = df[feature_cols]
    
    for col in cat_features:
        df[col] = df[col].astype('category')
    
    risk_prob = model.predict(df)[0]
    decision = "REJECT" if risk_prob >= optimal_threshold else "APPROVE"
    
    shap_values = explainer.shap_values(df)
    feature_impact = dict(zip(feature_cols, shap_values[0].tolist()))
    top_factors = sorted(feature_impact.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    return {
        "risk_probability": float(risk_prob),
        "decision": decision,
        "threshold_used": float(optimal_threshold),
        "top_contributing_factors": [{"feature": f, "impact": round(v, 4)} for f, v in top_factors]
    }