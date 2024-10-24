import pytest

from models.matches import Matches


def test_matches():
    matches = Matches.model_validate(Matches.Config.json_schema_extra["example"])
    assert isinstance(matches, Matches)


if __name__ == "__main__":
    pytest.main()
