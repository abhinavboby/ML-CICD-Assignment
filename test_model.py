from model import load_data, run, train_model


def test_data_loads():
    X, y = load_data()
    assert X.shape == (150, 4)
    assert len(y) == 150


def test_model_trains_and_predicts():
    X, y = load_data()
    model = train_model(X, y)
    preds = model.predict(X[:5])
    assert len(preds) == 5


def test_accuracy_above_threshold():
    _, accuracy = run()
    assert accuracy >= 0.90
