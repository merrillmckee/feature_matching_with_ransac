import pytest

from models.image_and_features import Feature, ImageAndFeatures


def test_feature():
    feature = Feature.model_validate(Feature.Config.json_schema_extra["example"])
    assert isinstance(feature, Feature)

def test_feature_no_descriptor():
    feature = Feature.model_validate(Feature.Config.json_schema_extra["example"])
    feature.descriptor = None
    assert isinstance(feature, Feature)

def test_feature_invalid_descriptor():
    example = Feature.Config.json_schema_extra["example"].copy()
    example["descriptor"] = [1.234] * 128  # good
    assert isinstance(Feature.model_validate(example), Feature)

    example["descriptor"] = 1.234
    with pytest.raises(ValueError):
        _ = Feature.model_validate(example)

    example["descriptor"] = [1.234]
    with pytest.raises(ValueError):
        _ = Feature.model_validate(example)

    example["descriptor"] = [1.234] * 127
    with pytest.raises(ValueError):
        _ = Feature.model_validate(example)


def test_image_and_features():
    image_and_features = ImageAndFeatures.model_validate(ImageAndFeatures.Config.json_schema_extra["example"])
    assert isinstance(image_and_features, ImageAndFeatures)


if __name__ == "__main__":
    pytest.main()
