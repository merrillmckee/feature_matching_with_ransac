import pytest

from src.models.image_and_features import Feature, ImageAndFeatures


def test_feature():
    feature = Feature.model_validate(Feature.Config.json_schema_extra["example"])
    assert isinstance(feature, Feature)


def test_image_and_features():
    image_and_features = ImageAndFeatures.model_validate(ImageAndFeatures.Config.json_schema_extra["example"])
    assert isinstance(image_and_features, ImageAndFeatures)


if __name__ == "__main__":
    pytest.main()
