import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Firstly, I want to compile all my processes for imputing 'CCA' into a function.
def impute_cca(df):
    # Let's define the features by which we want to predict the null 'CCA' values
    features = ["gender","learning_style","attendance_rate","hours_per_week","n_male","n_female","direct_admission","tuition","number_of_siblings", "sleep_time","wake_time"]

    # known = rows where 'CCA' is not null, unknown = rows where 'CCA' is null
    known = df[df["CCA"].notnull()]
    unknown = df[df["CCA"].isnull()]

    # This block is a safety measure; in case there are no null values under 'CCA' (in other words, unknown = 0), the original df will be returned
    # Otherwise, you may run into unnecessary errors later down the pipeline
    if unknown.empty:
        return df

    # Let's separate our categorical and numerical features
    cat_cols = known[features].select_dtypes(include=["object"]).columns.tolist()
    num_cols = known[features].select_dtypes(include=["int64","float64"]).columns.tolist()

    # Now, let's one-hot encode the X-variables for both the known and unknown sets
    X_cat_known = pd.get_dummies(known[cat_cols])
    X_cat_unknown = pd.get_dummies(unknown[cat_cols])
    # This line ensures that the known and unknown sets mirror each others' structure
    # In case there are some categories under some columns that are not present in both sets. 
    X_cat_unknown = X_cat_unknown.reindex(columns=X_cat_known.columns, fill_value=0)

    # Let's concat our encoded X-variables with the numerical columns (do for both known and unknown sets)
    X_known = pd.concat([known[num_cols].reset_index(drop=True),
                         X_cat_known.reset_index(drop=True)], axis=1)
    X_unknown = pd.concat([unknown[num_cols].reset_index(drop=True),
                           X_cat_unknown.reset_index(drop=True)], axis=1)

    # Let's define our y-variable
    y_known = known["CCA"]

    # Fitting a random forests model with X_known and y_known, before predicting using X_unknown 
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_known, y_known)
    preds = rf.predict(X_unknown)

    # Finally, we impute all the null values with our predictions
    df.loc[df["CCA"].isnull(), "CCA"] = preds
    return df


# Secondly, I want to standardise the formatting of 'tuition'
def standardise_tuition(x):
    mapping = {"Yes": "yes", "No": "no", "Y": "yes", "N": "no"}
    # .get() method parameters: key(if the key is found, corresponding value will be returned), default(if key not found, return this value)
    # Since our default = x, the original value will be returned if the key (x) cannot be found in mapping
    return mapping.get(x, x)

def fix_tuition(df):
    df["tuition"] = df["tuition"].apply(standardise_tuition)
    return df


# Thirdly, I want to create 'failed', our actual y-variable
def create_failed(df):
    """Create target variable 'failed' from final_test."""
    df["failed"] = df["final_test"].apply(lambda x: 0.0 if x >= 50 else 1.0)
    df = df.drop("final_test", axis=1)
    return df


# Fourth, I want to address the 'sleep' columns
# Re-engineering 'sleep_time' and 'wake_time'
def sleep_convert(x):
    if pd.isnull(x):
        return None
    if len(x) == 5:  # e.g. 23:45
        hrs = int(x[0:2])
        mins = int(x[3:5]) / 60
        return hrs + mins
    if len(x) == 4:  # e.g. 9:30
        hrs = int(x[0:1])
        mins = int(x[2:4]) / 60
        return hrs + mins
    # For safety: return none if none of the conditions above are fulfilled
    return None

# Creating 'sleep_dur'
def sleep_time_dur(x):
    if pd.isnull(x):
        return None
    if x > 10:
        return 24 - x
    if x < 10:
        return -x
    # For safety: return none if none of the conditions above are fulfilled
    return None

# Applying the two functions above to the respective columns/dataframe
def engineer_sleep_features(df):
    df["sleep_time"] = df["sleep_time"].apply(sleep_convert)
    df["wake_time"] = df["wake_time"].apply(sleep_convert)
    df["sleep_dur"] = df["sleep_time"].apply(sleep_time_dur) + df["wake_time"]
    # Deleting 'sleep_time' and 'wake_time' after engineering is complete
    df = df.drop(columns=["sleep_time", "wake_time"])
    return df


# Lastly, I want to standardise our categorical features
# 'categorical_features' can be found in the YAML doc as [columns][categorical]
def standardise_categoricals(df, categorical_features):
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].str.lower()
    return df
