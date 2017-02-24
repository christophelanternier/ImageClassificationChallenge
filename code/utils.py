import numpy as np
from numpy import power, log

def convolution_2D(image, filter):
    if image.shape != filter.shape:
        raise Exception("Image and filter shape must be the same")
    zero_padded_image = np.zeros((2 * image.shape[0], 2 * image.shape[1]))
    zero_padded_image[0:image.shape[0], 0:image.shape[1]] = image
    zero_padded_filter = np.zeros((2 * filter.shape[0], 2 * filter.shape[1]))
    zero_padded_filter[0:filter.shape[0], 0:filter.shape[1]] = filter

    filter_fft = np.fft.fft2(zero_padded_filter)
    image_fft = np.fft.fft2(zero_padded_image)

    return np.fft.ifft2(image_fft * filter_fft)[0:image.shape[0], 0:image.shape[1]]

def separate_RGB_images(image, size=1024):
    return image[:size], image[size:2*size], image[2*size:3*size]

def haar_wavelets_2D(size=32, return_type='vector'):
    n_mother_wavelets = 2
    n_scales = int(log(size) / log(2))

    wavelets = []
    for n_scale in range(n_scales):
        scale_wavelets = []
        scale = power(2, n_scale + 1)
        normalize_constant = 1.0 / power(scale, 2)

        for i in range(size / scale):
            I = i * scale
            for j in range(size / scale):
                J = j * scale
                horizontal_wavelet = np.zeros((size, size))
                horizontal_wavelet[I:I+scale,J:J+scale/2] = normalize_constant
                horizontal_wavelet[I:I+scale,J+scale/2:J+scale] = -normalize_constant

                vertical_wavelet = np.zeros((size, size))
                vertical_wavelet[I:I+scale/2,J:J+scale] = normalize_constant
                vertical_wavelet[I+scale/2:I+scale,J:J+scale] = -normalize_constant

                if return_type == 'vector':
                    scale_wavelets.append(horizontal_wavelet.reshape(size**2))
                    scale_wavelets.append(vertical_wavelet.reshape(size**2))
                else:
                    scale_wavelets.append(horizontal_wavelet)
                    scale_wavelets.append(vertical_wavelet)
        wavelets.append(scale_wavelets)

    return np.array(wavelets)
