import pandas as pd

def preprocess_data(train_df, test_df, target_col="SalePrice"):
    # Drop rows with missing target values
    train_df = train_df.dropna(subset=[target_col])

    # Store the target and drop ID
    y = train_df[target_col]
    train_df = train_df.drop([target_col, "Id"], axis=1)
    test_ids = test_df["Id"]
    test_df = test_df.drop("Id", axis=1)

    # Concatenate for consistent preprocessing
    full_data = pd.concat([train_df, test_df], axis=0)

    # Handle missing values (simple strategy)
    full_data = full_data.fillna(full_data.median(numeric_only=True))

    # Convert categorical to dummy vars
    full_data = pd.get_dummies(full_data)

    # Align train/test
    X_train = full_data.iloc[:len(y), :]
    X_test = full_data.iloc[len(y):, :]

    return X_train, y, X_test, test_ids
