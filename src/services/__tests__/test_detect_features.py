import numpy as np
import numpy.testing as npt
import pytest
from cv2 import Feature2D

from models.image_and_features import ImageAndFeatures
from services.detect_features import create_feature_detector, detect_features


def test_create_feature_detector():
    detector = create_feature_detector("SIFT")
    assert isinstance(detector, Feature2D)

    detector = create_feature_detector("KAZE")
    assert isinstance(detector, Feature2D)

    with pytest.raises(NotImplementedError):
        create_feature_detector("SURF")


def test_detect_features_black_image():
    gray = np.zeros(shape=(30, 40, 3), dtype=np.uint8) + 120
    image_and_features = detect_features(gray)
    assert isinstance(image_and_features, ImageAndFeatures)
    assert len(image_and_features.features) == 0
    npt.assert_array_equal(gray, image_and_features.image)


def test_detect_features_white_block():
    white_block = np.zeros(shape=(30, 40, 3), dtype=np.uint8)
    white_block[10:20, 10:30, :] = 255
    image_and_features = detect_features(white_block)
    assert isinstance(image_and_features, ImageAndFeatures)
    assert len(image_and_features.features) > 0
    npt.assert_array_equal(white_block, image_and_features.image)


if __name__ == "__main__":
    pytest.main()
