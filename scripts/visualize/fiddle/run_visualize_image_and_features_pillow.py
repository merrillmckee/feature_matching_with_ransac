import numpy as np

from scripts.visualize.visualize import visualize_image_and_features_pillow
from src.models.image_and_features import ImageAndFeatures


if __name__ == "__main__":
    image_and_features_ = ImageAndFeatures.model_validate(ImageAndFeatures.Config.json_schema_extra["example"])
    image_and_features_.image = np.zeros(shape=(360, 480, 3), dtype=np.uint8) + 100  # gray image
    visualize_image_and_features_pillow(image_and_features_)
