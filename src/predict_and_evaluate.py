from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# This function will predict y using X_test, which was returned from the train_model function
def evaluate_model(model, X_test, y_test):
    # Predicting y
    preds = model.predict(X_test)

    # accuracy, precision, recall
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)

    # confusion matrix
    cm = confusion_matrix(y_test, preds)
    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "confusion_matrix": cm}
    return metrics