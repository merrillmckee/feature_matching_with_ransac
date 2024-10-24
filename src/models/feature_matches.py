from pydantic import BaseModel, Field
from models.image_and_features import ImageAndFeatures


class FeatureMatches(BaseModel):
    """
    Describes image features matches between a pair of images
    """
    image_and_features_1: ImageAndFeatures = Field(
        description="Image and features for image 1",
    )
    image_and_features_2: ImageAndFeatures = Field(
        description="Image and features for image 1",
    )
    matches: list[tuple[int, int]] = Field(
        description="List of image pair match's feature indices between image 1 and image 2",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "image_and_features_1": ImageAndFeatures.Config.json_schema_extra["example"],
                "image_and_features_2": ImageAndFeatures.Config.json_schema_extra["example"],
                "matches": [[0, 0], [1, 1]],
            }
        }
