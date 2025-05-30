from quantem.imaging.drift import DriftCorrection
import numpy as np
import pytest

def test_init_with_one_image_provided():
    # Provide only one image, expect to raise ValueError
    img_data = np.random.random((10, 10))
    expected_error_msg = "DriftCorrection currently requires at least a pair of images to initialize."
    with pytest.raises(ValueError, 
                       match=expected_error_msg):
        DriftCorrection.from_data(
            images=[img_data],
            scan_direction_degrees=[0]
        )
