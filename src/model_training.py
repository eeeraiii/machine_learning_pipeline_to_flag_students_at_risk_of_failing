import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Now, I will train the logistic regression model using our processed dataset
def train_model(df, target="failed", model_params=None, save_path="model.pkl"):
    # .get_dummies is an efficient way to one-hot encode the categorical features in the dataframe as it will ignore numerical features
    X = pd.get_dummies(df.drop(columns=[target]))
    y = df[target]

    # stratify will ensure that the proportion of classes in y is maintained in both training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # The '**' will unpack the dictionary fed into the model
    model = LogisticRegression(**model_params)
    model.fit(X_train, y_train)

    # joblib is a Python library which saves Python objects to a file path
    # .dump() helps you dump the trained model somewhere?
    # You can load the model again later without needing to retrain it!
    joblib.dump(model, save_path)
    return model, X_test, y_test
