import numpy as np

from models.feature_matches import FeatureMatches
from models.image_transforms import Affine
from scripts.visualize.visualize import visualize_image_transforms


if __name__ == "__main__":
    feature_matches = FeatureMatches.model_validate(FeatureMatches.Config.json_schema_extra["example"])
    feature_matches.image_and_features_1.image = np.zeros(shape=(360, 480, 3), dtype=np.uint8) + 100  # gray image
    feature_matches.image_and_features_2.image = np.zeros(shape=(360, 480, 3), dtype=np.uint8) + 120  # gray image
    transform = Affine(
        # case 1
        # rotation of 10 CW about !origin!
        # transform=np.array([
        #     [0.9855, -0.169, 0],
        #     [0.169, 0.9855, 0],
        #     [0, 0, 1],
        # ], dtype=float)

        # case 2
        # scale of 1.5 from !origin!
        # transform=np.array([
        #     [1.5, 0, 0],
        #     [0, 1.5, 0],
        #     [0, 0, 1],
        # ], dtype=float)

        # case 3
        # random-ish points; parallel lines appear to hold
        transform=np.array([
            [1.5, -0.9, 15],
            [1.4, 0.8, -10],
            [0, 0, 1],
        ], dtype=float)
    )
    visualize_image_transforms(feature_matches, transform)
