import numpy as np
from numpy import power, log, exp, cos, sin, sqrt
from scipy.signal import fftconvolve

# Important constants
images_side = 32
nbpixels = images_side**2

def recenter(features):
    centered_features = np.copy(features)
    mean_feature = np.mean(features, axis=0)

    for i in range(features.shape[0]):
        centered_features[i,:] = features[i,:] - mean_feature

    return mean_feature, centered_features

def parties(card, n):
    #TODO I need comments from Louis
    ans = []

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
        if len(s) == card:
            ans.append(s)
        i += 1
    return ans

def separate_RGB_images(image, size=1024):
    return image[:size], image[size:2*size], image[2*size:3*size]

def get_rgb_array(data_line):
    rgb = separate_RGB_images(data_line)
    rgbArray = np.zeros((images_side,images_side, 3), 'uint8')
    for j,color in enumerate(rgb):
        color = color.reshape((images_side, images_side))
        rgbArray[..., j] = (color + color.min()) / (color.max() - color.min()) * 256

    return rgbArray

def get_gray_image(rgbArray):
    return 0.2989 * rgbArray[..., 0] + 0.5870 * rgbArray[..., 1] + 0.1140 * rgbArray[..., 2]


def average_and_subsample(image, size):
    N = image.shape[0] / size
    M = image.shape[1] / size
    averaged = np.zeros((N, M))

    for i in range(N):
        for j in range(M):
            averaged[i,j] = np.mean(image[i*size:(i+1)*size, j*size:(j+1)*size])

    return averaged

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
            wavelet[:] /= sqrt(np.sum(wavelet * wavelet))

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
    #TODO I don't get it, I need comments from Louis
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
