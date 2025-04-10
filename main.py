from src.data_loader import load_data
from src.preprocess import preprocess_data
from src.model import train_model, evaluate_model
from src.predict import make_predictions

# Load
train_df, test_df = load_data()

# Preprocess
X_train, y_train, X_test, test_ids = preprocess_data(train_df, test_df)

# Train & Evaluate
model = train_model(X_train, y_train)
mse, r2 = evaluate_model(model, X_train, y_train)
print(f"📉 Mean Squared Error: {mse:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# Predict & Save
make_predictions(model, X_test, test_ids)
