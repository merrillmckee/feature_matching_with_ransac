import numpy as np
import pytest

from models.image_transforms import Translation, Affine, Homography


def test_check_translation_transform():
    transform_example = Translation.Config.json_schema_extra["example"]

    # example
    translation = Translation.model_validate(transform_example)
    assert isinstance(translation, Translation)

    # identify matrix
    transform_example["transform"] = np.eye(N=3, dtype=float)
    translation = Translation.model_validate(transform_example)
    assert isinstance(translation, Translation)

    # invalid shape
    transform_example["transform"] = np.zeros(shape=(4, 5), dtype=float)
    with pytest.raises(ValueError):
        _ = Translation.model_validate(transform_example)

    # scaling not allowed
    transform_example["transform"] = np.eye(N=3, dtype=float) * 2.0
    with pytest.raises(ValueError):
        _ = Translation.model_validate(transform_example)

    # rotation not allowed
    transform_example["transform"] = np.array([[1, 0.5, 0], [0.86, 1, 0], [0, 0, 1]], dtype=float)
    with pytest.raises(ValueError):
        _ = Translation.model_validate(transform_example)


def test_check_affine_transform():
    transform_example = Affine.Config.json_schema_extra["example"]

    # example
    affine = Affine.model_validate(transform_example)
    assert isinstance(affine, Affine)

    # identify matrix
    transform_example["transform"] = np.eye(N=3, dtype=float)
    affine = Affine.model_validate(transform_example)
    assert isinstance(affine, Affine)

    # invalid shape
    transform_example["transform"] = np.zeros(shape=(4, 5), dtype=float)
    with pytest.raises(ValueError):
        _ = Affine.model_validate(transform_example)

    # 3rd row must be [0, 0, 1]
    transform_example["transform"] = np.array([[1, 0.5, 1], [0.86, 1, 2], [3, 4, 5]], dtype=float)
    with pytest.raises(ValueError):
        _ = Affine.model_validate(transform_example)


def test_check_homography():
    transform_example = Homography.Config.json_schema_extra["example"]

    # example
    homography = Homography.model_validate(transform_example)
    assert isinstance(homography, Homography)

    # identify matrix
    transform_example["transform"] = np.eye(N=3, dtype=float)
    homography = Homography.model_validate(transform_example)
    assert isinstance(homography, Homography)

    # invalid shape
    transform_example["transform"] = np.zeros(shape=(4, 5), dtype=float)
    with pytest.raises(ValueError):
        _ = Homography.model_validate(transform_example)

    # value at 3rd row, 3rd column must be zero
    transform_example["transform"] = np.array([[1, 0.5, 1], [0.86, 1, 2], [3, 4, 5]], dtype=float)
    with pytest.raises(ValueError):
        _ = Homography.model_validate(transform_example)


if __name__ == "__main__":
    pytest.main()
