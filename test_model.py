"""Automated tests for the Iris machine-learning model."""
import numpy as np
from model import train_model


def test_model_trains_and_returns_predictions():
    model, X_test, y_test, predictions, accuracy = train_model()
    assert model is not None
    assert len(predictions) == len(y_test)
    assert len(X_test) == len(y_test)
    assert 0.0 <= accuracy <= 1.0


def test_predictions_are_valid_iris_classes():
    _, _, _, predictions, _ = train_model()
    assert set(np.asarray(predictions).tolist()).issubset({0, 1, 2})


def test_model_accuracy_meets_basic_threshold():
    _, _, _, _, accuracy = train_model()
    assert accuracy >= 0.80
