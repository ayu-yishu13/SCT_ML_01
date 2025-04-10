import pandas as pd

def make_predictions(model, X_test, test_ids, filename="submission.csv"):
    preds = model.predict(X_test)
    submission = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": preds
    })
    submission.to_csv(filename, index=False)
    print("✅ submission.csv saved successfully!")
