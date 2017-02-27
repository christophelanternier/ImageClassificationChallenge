import numpy as np
from numpy import power, log, exp, cos, sin, sqrt
from scipy.signal import fftconvolve

def parties(card, n):
    ensemble = range(1, n+1)
    parties = []

    i = 0
    i_max = 2**n

    while i < i_max:
        s = []
        j = 0
        j_max = n
        while j < j_max:
            if (i>>j)&1 == 1:
                s.append(j+1)
            j += 1
        if (len(s) == card):
            parties.append(s)
        i += 1
    return parties

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


def generate_2D_wavelets(size, type='gabor'):
    shape = (size, size)

    horizontal_wavelet = np.zeros(shape)
    vertical_wavelet = np.zeros(shape)
    diagonal_wavelet_1 = np.zeros(shape)
    diagonal_wavelet_2 = np.zeros(shape)


    if type == 'gabor':
        freq = 1.0 / size
        c_x = size / 2
        c_y = size / 2
        for i in range(size):
            for j in range(size):
                y = (i + 0.5)
                x = (j + 0.5)
                shape = exp(-(power(x - c_x, 2) + power(y - c_y, 2)) * freq)

                vertical_wavelet[i,j] = shape * sin(2 * np.pi * y * freq)
                horizontal_wavelet[i,j] = shape * sin(2 * np.pi * x * freq)
                diagonal_wavelet_1[i,j] = shape * sin(2 * np.pi * (x + y - size) * freq)
                diagonal_wavelet_2[i,j] = shape * sin(2 * np.pi * (y - x) * freq)

        wavelets = [vertical_wavelet, horizontal_wavelet, diagonal_wavelet_1, diagonal_wavelet_2]

        for wavelet in wavelets:
            wavelet = wavelet / sqrt(np.sum(wavelet * wavelet))
    elif type == 'haar':
        normalize_constant = 1.0 / size**2

        horizontal_wavelet[0:size,0:size/2] = normalize_constant
        horizontal_wavelet[0:size,size/2:size] = -normalize_constant

        vertical_wavelet = horizontal_wavelet.T

        for i in range(size):
            for j in range(size):
                if (i < j) or (i == j and i % 2 == 0):
                    diagonal_wavelet_1[i,j] = normalize_constant
                else:
                    diagonal_wavelet_1[i,j] = -normalize_constant

                if (i + j < size - 1) or (i + j  == size - 1 and i % 2 == 0):
                    diagonal_wavelet_2[i,j] = normalize_constant
                else:
                    diagonal_wavelet_2[i,j] = -normalize_constant

        wavelets = [vertical_wavelet, horizontal_wavelet, diagonal_wavelet_1, diagonal_wavelet_2]
    else:
        raise Exception('Unknown wavelet type ' + str(type))

    return wavelets

def scattering_transform(image, order, maximum_scale, wavelet_type='gabor'):
    wavelets_banks = []
    for p in parties(order, maximum_scale):
        wavelets_bank = []
        for j in p:
            wavelet_size = power(2, j)
            wavelets_bank.append(generate_2D_wavelets(wavelet_size, type=wavelet_type))
        wavelets_banks.append(wavelets_bank)

    subsample_interval = power(2, maximum_scale)
    subsampled_features = []

    for img in separate_RGB_images(image):
        img = img.reshape((32,32))

        for wavelets_bank in wavelets_banks:
            features = [[img]]
            for wavelets in wavelets_bank:
                new_features = []

                for feature in features[-1]:
                    for wavelet in wavelets:
                        new_features.append(np.abs(fftconvolve(feature, wavelet, mode='same')))

                features.append(new_features)

            for i in range(len(features)):
                features_scale_n = features[i]
                for feature in features_scale_n:
                    subsampled_features.append(average_and_subsample(feature, subsample_interval).ravel())

    return np.concatenate(subsampled_features)
