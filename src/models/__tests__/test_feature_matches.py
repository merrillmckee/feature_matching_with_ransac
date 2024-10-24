import pytest

from models.feature_matches import FeatureMatches


def test_matches():
    matches = FeatureMatches.model_validate(FeatureMatches.Config.json_schema_extra["example"])
    assert isinstance(matches, FeatureMatches)


if __name__ == "__main__":
    pytest.main()
