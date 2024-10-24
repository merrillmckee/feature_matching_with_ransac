from PIL import Image, ImageDraw
from models.image_and_features import ImageAndFeatures


def visualize_image_and_features_pillow(image_and_features: ImageAndFeatures):
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
        x, y = feature.x, feature.y
        radius = max(2.0, feature.scale)
        x0, x1 = x - radius, x + radius
        y0, y1 = y - radius, y + radius
        draw.ellipse(xy=(x0, y0, x1, y1), fill=None, outline='yellow')
    image.show("title")
