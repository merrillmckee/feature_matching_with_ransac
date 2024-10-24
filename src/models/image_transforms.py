import numpy as np
from abc import ABC

from numpydantic import NDArray, Shape
from pydantic import BaseModel, Field, field_validator


class ImageTransform(ABC):
    # I'd like to promote field "transform" from the derived classes to this
    # base class but ran into issues with the pydantic validators. For now,
    # leaving "transform" in the derived classes.

    @classmethod
    def apply_transform(cls, points: NDArray[Shape["*, 2"], np.floating], transform: "ImageTransform") \
        -> NDArray[Shape["*, 2"], np.floating]:
        n = points.shape[0]
        points_3d = np.hstack((points, np.ones((n, 1))))
        xform_points_3d = np.matmul(points_3d, transform.transform.T)
        return xform_points_3d[:, :2]


class Translation(BaseModel, ImageTransform):
    transform: NDArray[Shape["3, 3"], np.floating] = Field(
        description="(3, 3) array to transform points from image 1 to image 2"
    )

    @field_validator("transform")
    def check_translation_transform(cls, value: NDArray[Shape["3, 3"], np.floating]) \
            -> NDArray[Shape["3, 3"], np.floating]:
        if value.shape != (3, 3):
            raise ValueError(f"Invalid transform shape: {value.shape}")
        if np.any(value.diagonal() != 1.0):
            raise ValueError(f"Invalid translation transform: {value}")
        if value[0, 1] != 0.0 or value[1, 0] != 0.0 or np.any(value[2, :2] != 0.0):
            raise ValueError(f"Invalid translation transform: {value}")
        return value

    class Config:
        json_schema_extra = {
            "example": {
                "transform": np.array(
                    object=[[1.0, 0.0, 30.0],
                            [0.0, 1.0, 50.0],
                            [0.0, 0.0, 1.0]],
                    dtype=float,
                ),
            }
        }


class Affine(BaseModel, ImageTransform):
    transform: NDArray[Shape["3, 3"], np.floating] = Field(
        description="(3, 3) array to transform points from image 1 to image 2"
    )

    @field_validator("transform")
    def check_affine_transform(cls, value: NDArray[Shape["3, 3"], np.floating]) \
            -> NDArray[Shape["3, 3"], np.floating]:
        if value.shape != (3, 3):
            raise ValueError(f"Invalid transform shape: {value.shape}")
        if np.any(value[2, :] != np.array([0.0, 0.0, 1.0], dtype=float)):
            raise ValueError(f"Invalid affine transform: {value}")
        return value

    class Config:
        json_schema_extra = {
            "example": {
                "transform": np.array(
                    object=[[1.1, -0.5, 30.0],
                            [0.86, 1.1, 50.0],
                            [0.0, 0.0, 1.0]],
                    dtype=float,
                ),
            }
        }


class Homography(BaseModel, ImageTransform):
    transform: NDArray[Shape["3, 3"], np.floating] = Field(
        description="(3, 3) array to transform points from image 1 to image 2"
    )

    @field_validator("transform")
    def check_homography(cls, value: NDArray[Shape["3, 3"], np.floating]) \
            -> NDArray[Shape["3, 3"], np.floating]:
        if value.shape != (3, 3):
            raise ValueError(f"Invalid transform shape: {value.shape}")
        if value[2, 2] != 1.0:
            raise ValueError(f"Invalid homography: {value}")
        return value

    class Config:
        json_schema_extra = {
            "example": {
                "transform": np.array(
                    object=[[1.1, -0.5, 30.0],
                            [0.86, 1.1, 50.0],
                            [0.9, 0.8, 1.0]],
                    dtype=float,
                ),
            }
        }
