import numpy as np
from numpydantic import NDArray, Shape
from pydantic import BaseModel, Field, field_validator


class Translation(BaseModel):
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


class Affine(BaseModel):
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


class Homography(BaseModel):
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
