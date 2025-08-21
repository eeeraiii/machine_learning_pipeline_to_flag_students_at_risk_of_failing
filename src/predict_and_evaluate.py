from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# This function will predict 'failed'
def get_predictions(model, X_test, y_test):
    # Predicting y
    preds = model.predict(X_test)
    predict_df = X_test.copy()
    predict_df['actual_failed'] = y_test
    predict_df['predicted_failed'] = preds
    return predict_df

# This function will evaluate our model
def evaluate_model(predict_df):
    # Predicting y, but only for reference
    actual = predict_df['actual_failed']
    predicted = predict_df['predicted_failed']
    # accuracy, precision, recall
    acc = accuracy_score(actual, predicted)
    prec = precision_score(actual, predicted)
    rec = recall_score(actual, predicted)

    # confusion matrix
    cm = confusion_matrix(actual, predicted)
    metrics = {
        "accuracy": acc, 
        "precision": prec, 
        "recall": rec, 
        "confusion_matrix": cm,
    }
    return metrics