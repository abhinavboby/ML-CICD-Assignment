"""Simple ML model: Logistic Regression on the Iris dataset."""
import json

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

MODEL_NAME = "Logistic Regression"
DATASET_NAME = "Iris"


def load_data():
    """Load the Iris dataset."""
    data = load_iris()
    return data.data, data.target


def train_model(X_train, y_train):
    """Train the model."""
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model


def run():
    """Load data, train, predict, and return (model, accuracy)."""
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, accuracy


if __name__ == "__main__":
    _, acc = run()
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Accuracy: {acc * 100:.2f}%")

    with open("metrics.json", "w") as f:
        json.dump(
            {"model": MODEL_NAME, "dataset": DATASET_NAME, "accuracy": round(acc * 100, 2)},
            f,
        )
