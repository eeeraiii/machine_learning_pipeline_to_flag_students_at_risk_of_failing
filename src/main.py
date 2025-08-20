from data_loading_preprocessing import load_config, load_data_from_db, clean_data
from feature_engineering import fix_tuition, engineer_sleep_features, impute_cca, create_failed, standardise_categoricals
from model_training import train_model
from predict_and_evaluate import evaluate_model
import pandas as pd

def main():
    # Loading in our YAML file as a Python dictionary we can reference
    config = load_config()

    # Loading and cleaning data (data_loading_preprocessing)
    df = load_data_from_db(config["data"]["db_path"], config["data"]["raw_csv"])
    df = pd.read_csv(config["data"]["raw_csv"])
    df = clean_data(df, config["columns"]["drop"])

    # Feature engineering (feature_engineering)
    df = impute_cca(df)
    df = fix_tuition(df)
    df = create_failed(df)
    df = engineer_sleep_features(df)
    df = standardise_categoricals(df, config["columns"]["categorical"])

    # Save cleaned dataset
    df.to_csv(config["data"]["cleaned_csv"], index=False)

    # Training model using logistic regression (model_training)
    model, X_test, y_test = train_model(
        df,
        target="failed",
        model_params=config["model"]["params"],
        save_path="model.pkl"
    )

    # Step 4: Predict using X_test and 
    results = evaluate_model(model, X_test, y_test)
    print(results)
    
if __name__ == "__main__":
    main()
