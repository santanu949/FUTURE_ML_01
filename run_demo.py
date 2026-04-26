import pandas as pd
from src.data.processor import DataGenerator, DataProcessor
from src.models.engine import ModelEngine

def run_enterprise_demo():
    print("="*50)
    print("   FUTURE FORECAST PRO - ENTERPRISE DEMO")
    print("="*50)
    
    # 1. Pipeline: Data Generation
    print("\n[1/4] INITIALIZING DATA PIPELINE...")
    df = DataGenerator.generate_sales_data(periods=350, noise_level=0.15)
    print(f"SUCCESS: Generated 350 days of synthetic sales data.")
    
    # 2. Pipeline: Advanced Processing
    print("\n[2/4] EXECUTING FEATURE ENGINEERING...")
    df_clean = DataProcessor.validate_data(df, 'ds', 'y')
    # Detecting outliers
    df_no_outliers = DataProcessor.handle_outliers(df_clean, 'y', method='isolation_forest')
    # Engineering lags & rolling stats
    df_final = DataProcessor.feature_engineering(df_no_outliers.set_index('ds'), 'y').reset_index()
    print(f"SUCCESS: Created features: {list(df_final.columns[-5:])}")
    
    # 3. Model Engine: Automated Benchmarking
    print("\n[3/4] STARTING MODEL BENCHMARKING (PROPHET VS XGBOOST)...")
    engine = ModelEngine(df_final)
    # This will perform Optuna tuning for Prophet and train XGBoost
    best_model, model_type = engine.compare_and_select()
    
    # 4. Insight Generation
    print("\n[4/4] GENERATING BUSINESS INSIGHTS...")
    print(f"SELECTED PRODUCTION MODEL: {model_type.upper()}")
    
    if hasattr(engine, 'feature_importance'):
        print("\nTOP PREDICTIVE FACTORS:")
        for feat, score in sorted(engine.feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]:
            print(f" - {feat.upper()}: {score*100:.1f}% Influence")
            
    print("\n" + "="*50)
    print("   ENTERPRISE FORECASTING SYSTEM IS ONLINE")
    print("="*50)

if __name__ == "__main__":
    run_enterprise_demo()
