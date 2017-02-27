import numpy as np
from numpy import power, log, exp, cos, sin, sqrt
from scipy.signal import fftconvolve

def convolution_2D(image, filter):
    if image.shape != filter.shape:
        raise Exception('Image (shape = '+ str(image.shape) + ') and filter (shape = ' + str(filter.shape) + ') must have same shape')
    zero_padded_image = np.zeros((2 * image.shape[0], 2 * image.shape[1]))
    zero_padded_image[0:image.shape[0], 0:image.shape[1]] = image
    zero_padded_filter = np.zeros((2 * filter.shape[0], 2 * filter.shape[1]))
    zero_padded_filter[0:filter.shape[0], 0:filter.shape[1]] = filter

    filter_fft = np.fft.fft2(zero_padded_filter)
    image_fft = np.fft.fft2(zero_padded_image)

    return np.fft.ifft2(image_fft * filter_fft)[0:image.shape[0], 0:image.shape[1]].real

def separate_RGB_images(image, size=1024):
    return image[:size], image[size:2*size], image[2*size:3*size]

def average_and_subsample(image, size):
    N = image.shape[0] / size
    M = image.shape[1] / size
    averaged = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            averaged[i,j] = np.mean(image[i*size:(i+1)*size, j*size:(j+1)*size])

    return averaged

def generate_haar_wavelets_2D(scale, size=32):
    normalize_constant = 1.0 / scale**2

    horizontal_wavelet = np.zeros((size, size))
    vertical_wavelet = np.zeros((size, size))
    diagonal_wavelet_1 = np.zeros((size, size))
    diagonal_wavelet_2 = np.zeros((size, size))

    horizontal_wavelet[0:scale,0:scale/2] = normalize_constant
    horizontal_wavelet[0:scale,scale/2:scale] = -normalize_constant

    vertical_wavelet = horizontal_wavelet.T

    for i in range(scale):
        for j in range(scale):
            if (i < j) or (i == j and i % 2 == 0):
                diagonal_wavelet_1[i,j] = normalize_constant
            else:
                diagonal_wavelet_1[i,j] = -normalize_constant

            if (i + j < scale - 1) or (i + j  == scale - 1 and i % 2 == 0):
                diagonal_wavelet_2[i,j] = normalize_constant
            else:
                diagonal_wavelet_2[i,j] = -normalize_constant

    return [vertical_wavelet, horizontal_wavelet, diagonal_wavelet_1, diagonal_wavelet_2]


def generate_gabor_wavelets_2D(scale, size=32):
    freq = 1.0 / scale

    horizontal_wavelet = np.zeros((size, size))
    vertical_wavelet = np.zeros((size, size))
    diagonal_wavelet_1 = np.zeros((size, size))
    diagonal_wavelet_2 = np.zeros((size, size))


    c_x = scale / 2
    c_y = scale / 2
    for i in range(scale):
        for j in range(scale):
            y = (i + 0.5)
            x = (j + 0.5)
            shape = exp(-(power(x - c_x, 2) + power(y - c_y, 2)) * freq)

            vertical_wavelet[i,j] = shape * sin(2 * np.pi * y * freq)
            horizontal_wavelet[i,j] = shape * sin(2 * np.pi * x * freq)
            diagonal_wavelet_1[i,j] = shape * sin(2 * np.pi * (x + y - scale) * freq)
            diagonal_wavelet_2[i,j] = shape * sin(2 * np.pi * (y - x) * freq)

    wavelets = [vertical_wavelet, horizontal_wavelet, diagonal_wavelet_1, diagonal_wavelet_2]

    for wavelet in wavelets:
        wavelet = wavelet / sqrt(np.sum(wavelet * wavelet))

    return wavelets

def scattering_transform(image, maximum_scale, wavelet_type='gabor'):
    scale_wavelets = []
    for j in range(maximum_scale):
        scale = power(2, (j+1))
        if wavelet_type == 'gabor':
            scale_wavelets.append(generate_gabor_wavelets_2D(scale, scale))
        else:
            scale_wavelets.append(generate_haar_wavelets_2D(scale, scale))

    # compute all features
    features = []

    for img in separate_RGB_images(image):
        img = img.reshape((32,32))

        features.append([img])
        for wavelets in scale_wavelets:
            new_features = []

            for feature in features[-1]:
                for wavelet in wavelets:
                    new_features.append(np.abs(fftconvolve(feature, wavelet, mode='same')))

    # compute subsampled features
    subsampled_features = []
    subsample_interval = power(2, maximum_scale)
    for i in range(len(features)):
        features_scale_n = features[i]
        for feature in features_scale_n:
            subsampled_features.append(average_and_subsample(feature, subsample_interval).ravel())

    return np.concatenate(subsampled_features)
