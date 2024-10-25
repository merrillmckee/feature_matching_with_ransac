import numpy as np
from numpydantic import NDArray
from PIL import Image


def separate_image_in_2(image: NDArray) -> (NDArray, NDArray):
    # separate an image into 2; artificially trim so halves are less lined up
    rows, cols, _ = image.shape
    rows_buffer = 20
    columns_buffer = 50
    target_rows = rows - rows_buffer
    target_cols = cols // 2 - columns_buffer
    image_1 = image[:target_rows, :target_cols, :]
    image_2 = image[-target_rows:, -target_cols:, :]
    assert image_1.shape == image_2.shape
    return image_1, image_2


if __name__ == "__main__":
    # image from url - https://i.sstatic.net/p11WT.jpg
    image_ = Image.open("images/p11WT.jpg")
    image_1_arr, image_2_arr = separate_image_in_2(np.array(image_))
    image_.close()

    image_1 = Image.fromarray(image_1_arr)
    image_2 = Image.fromarray(image_2_arr)
    # image_1.show("image 1")
    # image_2.show("image 2")
    image_1.save("images/moon_1.png")
    image_2.save("images/moon_2.png")
    image_1.close()
    image_2.close()
