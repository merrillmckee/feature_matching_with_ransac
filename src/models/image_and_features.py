import numpy as np
from numpydantic import NDArray, Shape
from pydantic import BaseModel, Field


class Feature(BaseModel):
    """
    Describes an image feature like SIFT or KAZE
    """
    x: float = Field(
        description="x-coordinate of feature in pixels",
    )
    y: float = Field(
        description="y-coordinate of feature in pixels",
    )
    scale: float = Field(
        description="scale of the feature in pixels",
        default=1.0,
    )
    rotation: float = Field(
        description="rotation angle, in degrees CCW from +x-axis",
        default=90.0,
    )

    class Config:
        json_schema_extra = {
            "example": {
                "x": 100,
                "y": 200,
                "scale": 1.0,
                "rotation": 90.0,
            }
        }


class ImageAndFeatures(BaseModel):
    """
    An image and its associated features
    """
    image: NDArray[Shape["* x, * y, 3 rgb"], np.uint8] = Field(
        description="(H, W, Ch) RGB image",
    )
    features: list[Feature] = Field(
        description="List of image features",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "image": np.array(
                    # (2, 3) RGB image of alternating black and blue pixels
                    object=[[[0, 0, 0], [0, 0, 255], [0, 0, 0]], [[0, 0, 255], [0, 0, 0], [0, 0, 255]]],
                    dtype=np.uint8,
                ),
                "features": [Feature.Config.json_schema_extra["example"]],
            }
        }