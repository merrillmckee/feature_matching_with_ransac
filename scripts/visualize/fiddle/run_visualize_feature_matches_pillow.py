import numpy as np

from models.feature_matches import FeatureMatches
from scripts.visualize.visualize import visualize_feature_matches_pillow


if __name__ == "__main__":
    feature_matches = FeatureMatches.model_validate(FeatureMatches.Config.json_schema_extra["example"])
    feature_matches.image_and_features_1.image = np.zeros(shape=(360, 480, 3), dtype=np.uint8) + 100  # gray image
    feature_matches.image_and_features_2.image = np.zeros(shape=(360, 480, 3), dtype=np.uint8) + 120  # gray image
    visualize_feature_matches_pillow(feature_matches)
