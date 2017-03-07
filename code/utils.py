import numpy as np
from numpy import power, exp, cos, sin, pi
from scipy.signal import fftconvolve

# Important constants
images_side = 32
nbpixels = images_side**2

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

def generate_2D_wavelets(size, type='gabor', n_rotations=4):
    shape = (size, size)

    wavelets = [np.zeros(shape) for k in range(n_rotations)]

    if type == 'gabor':
        freq = 1.0 / size
        c_x = size / 2
        c_y = size / 2
        sigma2 = (1.0 * size / 2)**2
        for i in range(size):
            for j in range(size):
                y = (i + 0.5)
                x = (j + 0.5)
                shape = exp(- 0.5 * (power(x - c_x, 2) + power(y - c_y, 2)) / sigma2)

                for gamma in range(n_rotations):
                    angle = pi * gamma / n_rotations
                    k_x = cos(angle)
                    k_y = sin(angle)
                    wavelets[gamma][i,j] = shape * sin(2 * pi * (k_x * (x - c_x) + k_y * (y - c_y)) * freq)

        for i in range(len(wavelets)):
            wavelets[i] /= np.linalg.norm(wavelets[i])

    elif type == 'haar':
        for i in range(size):
            for j in range(size):
                for gamma in range(n_rotations):
                    raise Exception("Not implemented yet")
    else:
        raise Exception('Unknown wavelet type ' + str(type))

    return wavelets

def scattering_transform(image, order, maximum_scale, wavelet_type='gabor', normalize_features=False):
    #TODO I don't get it, I need comments from Louis
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
