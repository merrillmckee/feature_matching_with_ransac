import numpy as np
from PIL import Image, ImageDraw
from models.image_and_features import ImageAndFeatures, Feature
from models.feature_matches import FeatureMatches


def feature_to_bounding_box(feature: Feature) -> tuple[float, float, float, float]:
    """
    Helper function to convert a feature to a display bounding box of the form (x0, y0, x1, y1)

    Parameters
    ----------
    feature:
        feature object with xy coordinates and scale

    Returns
    -------
    bounding_box:
        display bounding box of the form (x0, y0, x1, y1)
    """
    x, y = feature.x, feature.y
    radius = max(2.0, feature.scale)
    x0, x1 = x - radius, x + radius
    y0, y1 = y - radius, y + radius
    bounding_box = x0, y0, x1, y1
    return bounding_box


def visualize_image_and_features(image_and_features: ImageAndFeatures):
    """
    Displays an image with overlays for each feature

    Parameters
    ----------
    image_and_features:
        An ImageAndFeatures object to be visualized
    """
    image = Image.fromarray(image_and_features.image)
    draw = ImageDraw.Draw(image)
    for feature in image_and_features.features:
        bounding_box = feature_to_bounding_box(feature)
        draw.ellipse(xy=bounding_box, fill=None, outline='yellow')
    image.show()


def visualize_feature_matches(feature_matches: FeatureMatches):
    """
    Displays a pair of images with overlays for each feature match

    Note: for now, requires same size images

    Parameters
    ----------
    feature_matches:
        A Matches object to be visualized
    """
    image_1: np.typing.NDArray = feature_matches.image_and_features_1.image
    image_2: np.typing.NDArray = feature_matches.image_and_features_2.image
    if image_1.shape != image_2.shape:
        raise NotImplemented("Visualization of different sized images as feature matches not yet implemented")

    image_1_cols = image_1.shape[1]
    features_1 = feature_matches.image_and_features_1.features
    features_2 = feature_matches.image_and_features_2.features
    image_pair = Image.fromarray(np.hstack((image_1, image_2)))
    draw = ImageDraw.Draw(image_pair)
    for match in feature_matches.matches:
        feature_1 = features_1[match[0]]
        bounding_box_1 = feature_to_bounding_box(feature_1)
        draw.ellipse(xy=bounding_box_1, fill=None, outline='yellow')

        feature_2 = features_2[match[1]]
        x0, y0, x1, y1 = feature_to_bounding_box(feature_2)
        bounding_box_2 = x0 + image_1_cols, y0, x1 + image_1_cols, y1
        draw.ellipse(xy=bounding_box_2, fill=None, outline='yellow')

        draw.line(xy=(feature_1.x, feature_1.y, feature_2.x + image_1_cols, feature_2.y), fill='blue')
    image_pair.show()
