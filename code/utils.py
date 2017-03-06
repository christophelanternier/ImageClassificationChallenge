import numpy as np
from numpy import power, log, exp, cos, sin, sqrt
from scipy.signal import fftconvolve

def compute_distances_to_mean_features(features, labels):
    distances = np.zeros(features.shape[0], 10)
    mean_features = np.zeros(10, features.shape[1])
    n_nearer_mean = np.zeros(features.shape[0])

    for label in range(10):
        label_indices = np.where(labels == label)[0]
        label_features = features[label_indices,:]
        mean_features[label,:] = np.mean(label_features, axis=0)

    for i in range(features.shape[0]):
        distance_to_true_label = np.linalg.norm(features[i,:] - label_features[labels[i],:])
        for label in range(10):
            distances[i, label] = np.linalg.norm(features[i,:] - label_features[label,:])
            if (distances[i, label] < distance_to_true_label):
                n_nearer_mean[i] += 1

    return distances, n_nearer_mean

def normalize(features):
    max_norm = np.max(np.linalg.norm(features, axis=1))
    max_norm_inv = 1.0 / max_norm
    features *= max_norm_inv

    return features

def compute_projection_on_affine_space(vector, point, orthogonal_space_basis):
    relative_vector = vector - point

    basis_coefficients = orthogonal_space_basis.dot(relative_vector)
    projection_on_orthogonal_space = np.sum(np.diag(basis_coefficients).dot(orthogonal_space_basis), axis=0)

    projection = vector
    ## WORK IN PROGRESS


def recenter(features):
    centered_features = np.copy(features)
    mean_feature = np.mean(features, axis=0)

    for i in range(features.shape[0]):
        centered_features[i,:] = features[i,:] - mean_feature

    return mean_feature, centered_features

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

def scattering_transform(image, order, maximum_scale, wavelet_type='gabor', normalize_features=True):
    wavelets_banks = []
    for p in parties(order, maximum_scale):
        wavelets_bank = []
        for j in p:
            wavelet_size = power(2, j)
            wavelets_bank.append(generate_2D_wavelets(wavelet_size, type=wavelet_type))
        wavelets_banks.append(wavelets_bank)

    subsample_interval = power(2, maximum_scale)
    output_features = []

    for img in separate_RGB_images(image):
        img = img.reshape((32,32))

        for wavelets_bank in wavelets_banks:
            # compute feature corresponding to a p
            features = [[img]]
            for wavelets in wavelets_bank:
                new_features = []

                for feature in features[-1]:
                    for wavelet in wavelets:
                        new_features.append(np.abs(fftconvolve(feature, wavelet, mode='same')))

                features.append(new_features)

            subsampled_features = []
            for i in range(len(features)):
                features_scale_n = features[i]
                for feature in features_scale_n:
                    subsampled_features.append(average_and_subsample(feature, subsample_interval).ravel())

            if normalize_features:
                normalized_features = normalize(np.concatenate(subsampled_features))
                output_features.append(normalized_features)
            else:
                output_features.append(np.concatenate(subsampled_features))

    return np.concatenate(output_features)
