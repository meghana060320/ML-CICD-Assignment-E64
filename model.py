"""Train and evaluate a Decision Tree classifier on the Iris dataset."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


def train_model():
    """Return a trained model, test data, predictions, and accuracy."""
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42,
        stratify=iris.target,
    )
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return model, X_test, y_test, predictions, accuracy


def main():
    iris = load_iris()
    _, X_test, y_test, predictions, accuracy = train_model()
    print("ML Model: Decision Tree Classifier")
    print(f"Dataset: {iris.DESCR.splitlines()[0].strip()}")
    print(f"Test samples: {len(y_test)}")
    print(f"Predictions: {predictions.tolist()}")
    print(f"Actual labels: {y_test.tolist()}")
    print(f"Model accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")


if __name__ == "__main__":
    main()
