import sqlite3
import pandas as pd
import yaml

# This function will take the YAML file and convert it to a Python dictionary. 
# We will later assign the function to a variable and can access the indices of the dictionary by calling the variable
def load_config(config_path="config.yaml"):
    # open(config_path,'r') opens the yaml file in read mode
    # enclosing the open() function in a with... as... format ensures the file is closed after the block is finished 
    with open(config_path, "r") as f:
        # .safe_load() turns the YAML text into a Python dictionary for easier reference in future
        return yaml.safe_load(f)

# This function will load the data from the SQLite database (the .db file) and convert it into a .csv file
def load_data_from_db(db_path, output_csv_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM score"
    df = pd.read_sql_query(query, conn)
    df.to_csv(output_csv_path, index=False)
    return df

''' This function will perform the following:
1. Drop duplicates according to 'student_id'
2. Drop rows where values under 'final_test' or 'attendance_rate' are null
3. Drop columns which I have determined would remain untouched during the cleaning/engineering process and don't need for modelling ("index", "age", "bag_color", "student_id", "mode_of_transport")
'''

def clean_data(df, drop_cols):
    # Process 1
    df = df.drop_duplicates(subset=["student_id"], keep="first")
    # Process 2
    df = df.dropna(axis=0, subset=["final_test", "attendance_rate"])
    # Process 3
    df = df.drop(columns=drop_cols, axis=1)
    return df
