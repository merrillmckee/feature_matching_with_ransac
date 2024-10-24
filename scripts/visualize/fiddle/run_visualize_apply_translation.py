import numpy as np

from models.feature_matches import FeatureMatches
from models.image_transforms import Translation
from scripts.visualize.visualize import visualize_image_transforms


if __name__ == "__main__":
    feature_matches = FeatureMatches.model_validate(FeatureMatches.Config.json_schema_extra["example"])
    feature_matches.image_and_features_1.image = np.zeros(shape=(360, 480, 3), dtype=np.uint8) + 100  # gray image
    feature_matches.image_and_features_2.image = np.zeros(shape=(360, 480, 3), dtype=np.uint8) + 120  # gray image
    transform = Translation(
        transform=np.array([
            [1, 0, 10],
            [0, 1, 40],
            [0, 0, 1],
        ], dtype=float)
    )
    visualize_image_transforms(feature_matches, transform)
