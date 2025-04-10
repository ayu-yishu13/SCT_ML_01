from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_train, y_train):
    preds = model.predict(X_train)
    mse = mean_squared_error(y_train, preds)
    r2 = r2_score(y_train, preds)
    return mse, r2
