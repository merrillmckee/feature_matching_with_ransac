import numpy as np
from cv2 import DMatch, FlannBasedMatcher

from models.feature_matches import FeatureMatches
from models.image_and_features import ImageAndFeatures


def match_features(
        image_and_features_1: ImageAndFeatures,
        image_and_features_2: ImageAndFeatures,
        ratio_threshold: float = 0.7,
) -> FeatureMatches:
    descriptors_1 = np.array(image_and_features_1.descriptors, np.float32)
    descriptors_2 = np.array(image_and_features_2.descriptors, np.float32)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = FlannBasedMatcher(index_params, search_params)
    raw_matches = flann.knnMatch(descriptors_1, descriptors_2, k=2)

    # Lowe's ratio test
    matches: list[DMatch] = []
    for match_1st_best, match_2nd_best in raw_matches:
        if match_1st_best.distance < ratio_threshold * match_2nd_best.distance:
            # best match has significantly lower distance/error than 2nd best match
            matches.append(match_1st_best)
    feature_matches = FeatureMatches(
        image_and_features_1=image_and_features_1,
        image_and_features_2=image_and_features_2,
        matches=[(match.queryIdx, match.trainIdx) for match in matches],
    )
    return feature_matches
