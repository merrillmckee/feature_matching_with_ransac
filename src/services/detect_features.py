import cv2
import numpy as np
from numpydantic import NDArray, Shape

from models.image_and_features import ImageAndFeatures, Feature


def create_feature_detector(algorithm: str) -> cv2.Feature2D:
    """
    Creates an OpenCV Feature2D instance with the chosen algorithm. Algorithms
    currently supported are {"SIFT", "KAZE"}.

    Parameters
    ----------
    algorithm:
        algorithm choice from {"SIFT", "KAZE"}

    Returns
    -------
    detector:
        OpenCV Feature2D instance
    """
    if algorithm.upper() == "SIFT":
        detector = cv2.SIFT_create(
            # nfeatures: int,
            # nOctaveLayers: int,
            # contrastThreshold: float,
            # edgeThreshold: float,
            # sigma: float,
            # descriptorType: int,
            # enable_precise_upscale: bool,
        )
    elif algorithm.upper() == "KAZE":
        detector = cv2.KAZE_create(
            # extended: bool = ...,
            # upright: bool = ...,
            # threshold: float = ...,
            # nOctaves: int = ...,
            # nOctaveLayers: int = ...,
            # diffusivity: KAZE_DiffusivityType,
        )
    else:
        raise NotImplementedError(f"Algorithm {algorithm} not implemented")
    return detector


def detect_features(
        image: NDArray[Shape["* h, * w, 3 rgb"], np.uint8],
        algorithm: str = "SIFT",
) -> ImageAndFeatures:
    """
    Detects features (keypoints) in an image

    Parameters
    ----------
    image:
        image of shape (H, W, 3)
    algorithm:
        algorithm choice from {"SIFT", "KAZE"}

    Returns
    -------
    image_and_features:
        an ImageAndFeatures instance
    """
    detector = create_feature_detector(algorithm)
    features_cv, descriptors = detector.detectAndCompute(image, mask=None)

    # DEBUG
    # out = cv2.drawKeypoints(image, features, outImage=None, color=(0, 255, 0))
    # cv2.imshow('Image with keypoints', out)
    # key = cv2.waitKey(0)
    # cv2.destroyAllWindows()

    features = [Feature(x=feat.pt[0], y=feat.pt[1], scale=feat.size / 2.0) for feat in features_cv]
    image_and_feature = ImageAndFeatures(
        image=image,
        features=features,
    )
    return image_and_feature
