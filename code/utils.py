import numpy as np

def 2D_convolution(image, filter):
    if image.shape != filter.shape:
        raise Exception("Image and filter shape must be the same")

    filter_fft = np.fft.fft2(filter)
    image_fft = np.fft.fft2(image)

    return np.fft.ifft2(image_fft * filter_fft)
