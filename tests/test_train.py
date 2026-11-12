from src.train import train_model


def test_training_produces_good_model():
    metadata = train_model()
    assert metadata["accuracy"] >= 0.9
    assert metadata["dataset"] == "Breast Cancer Wisconsin Diagnostic"
    assert metadata["classes"] == ["benign", "malignant"]
    assert len(metadata["features"]) == 30
