import numpy as np
from PIL import Image

from scripts.visualize.visualize import visualize_image_and_features, visualize_feature_matches
from services.detect_features import detect_features
from services.match_features import match_features


if __name__ == "__main__":
    algorithm = "SIFT"  # {"SIFT", "KAZE"}

    image = Image.open("../images/moon_1.png")
    image_and_features_1 = detect_features(np.array(image), algorithm=algorithm)
    # visualize_image_and_features(image_and_features_1)

    image = Image.open("../images/moon_2.png")
    image_and_features_2 = detect_features(np.array(image), algorithm=algorithm)
    # visualize_image_and_features(image_and_features_2)

    matches = match_features(image_and_features_1, image_and_features_2, ratio_threshold=0.2)
    visualize_feature_matches(matches)
