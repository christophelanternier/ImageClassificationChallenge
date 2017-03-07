import numpy as np
import pywt
import pywt.data
from utils import *
from scipy.signal import fftconvolve

def wavelet_transform(Xtr):
    """Return the dwt2 transform of the gray level images."""
    result = np.zeros((Xtr.shape[0], 972))

    for i in range(Xtr.shape[0]):
        rgbArray = get_rgb_array(Xtr[i])

        grayImage = get_gray_image(rgbArray)
        coeffs2 = pywt.dwt2(grayImage, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2

        result[i,:] = np.concatenate((LH.ravel(), HL.ravel(), HH.ravel()), axis=0)
    return result

def fourier_modulus_1D_kernel(images, signal='rows'):
    """Return the fourier transform of the rows/columns of the image."""

    fourier_1D_features_rows = np.zeros_like(images)
    fourier_1D_features_columns = np.zeros_like(images)

    for i in range(images.shape[0]):
        RGB = separate_RGB_images(images[i])

        for j, image in enumerate(RGB):
            rows = image
            columns = image.reshape((images_side,images_side)).T.ravel()

            fourier_1D_features_rows[i,j*nbpixels:(j+1)*nbpixels] = np.abs(np.fft.fft(rows))
            fourier_1D_features_columns[i,j*nbpixels:(j+1)*nbpixels] = np.abs(np.fft.fft(columns))

    if signal == 'rows':
        return fourier_1D_features_rows
    elif signal == 'columns':
        return fourier_1D_features_columns
    else:
        return np.concatenate([fourier_1D_features_rows, fourier_1D_features_columns])


def fourier_modulus_1D_kernel_2(images):
    """Return the fourier transform of each image seen as a 1D array."""
    n_images = images.shape[0]
    fourier_1D_features = np.zeros((n_images, 2*3*nbpixels))

    for i in range(n_images):
        RGB = separate_RGB_images(images[i])

        for j, image in enumerate(RGB):
            columns = image.reshape((32, 32)).T.ravel()
            fourier_1D_features[i, (2*j)*nbpixels: (2*j+1)*nbpixels] = np.abs(np.fft.fft(image))
            fourier_1D_features[i, (2*j+1)*nbpixels: (2*j+2)*nbpixels] = np.abs(np.fft.fft(columns))

    return fourier_1D_features


def fourier_modulus_2D_kernel(images):
    """Return the absolute values of the 2D Fourier transform of each image in the array images."""
    fourier_2D_modulus = np.zeros_like(images)

    for i in range(images.shape[0]):
        RGB = separate_RGB_images(images[i])

        for j, image in enumerate(RGB):
            fourier_2D_modulus[i, j*nbpixels: (j+1)*nbpixels] = np.abs(np.fft.fft2(image)).ravel()

    return fourier_2D_modulus


def fourier_phase_2D_kernel(images):
    """Return the phase of the 2D Fourier transform of each image in the array images."""
    fourier_2D_phase = np.zeros_like(images)

    for i in range(images.shape[0]):
        RGB = separate_RGB_images(images[i])

        for j, image in enumerate(RGB):
            fourier_2D_phase[i, j*nbpixels: (j+1)*nbpixels] = np.angle(np.fft.fft2(image)).ravel()

    return fourier_2D_phase


def scattering_kernel(images, order, scale, wavelet_type='gabor'):
    """Return the scattering transform of each image in the array images."""
    # TODO what is order and scale
    n_images = images.shape[0]
    scattering_transform_size = scattering_transform(images[0,:], order, scale, wavelet_type=wavelet_type).size
    scattering_features = np.zeros((n_images, scattering_transform_size))

    for i in range(images.shape[0]):
        image = images[i,:]
        scattering_features[i,:] = scattering_transform(image, order, scale, wavelet_type=wavelet_type)

    return scattering_features

def first_scattering_kernel(images, wavelet_type='gabor', normalize_features=True, image_subsample_size = 4, subsample_sizes=[2, 4], scales=[4, 8]):
    scale_wavelets = []
    for scale in scales:
        scale_wavelets.append(generate_2D_wavelets(scale, type=wavelet_type))

    subsampled_image_size = nbpixels / image_subsample_size**2

    # compute size of scattering transform
    feature_size = subsampled_image_size
    n_feature_maps = 1
    for i, (scale, subsample_size) in enumerate(zip(scales, subsample_sizes)):
        wavelets = scale_wavelets[i]
        n_feature_maps *= len(wavelets)
        feature_size += n_feature_maps * nbpixels / subsample_size**2

    scattering_features = np.zeros((images.shape[0], 3 * feature_size))

    for i in range(images.shape[0]):
        RGB = separate_RGB_images(images[i])

        end_index = 0
        for j, image in enumerate(RGB):
            image = image.reshape((32,32))

            features = [[image]]

            # compute all features
            for k, wavelets in enumerate(scale_wavelets):
                new_features = []
                for wavelet in wavelets:
                    for feature in features[-1]:
                        new_features.append(np.abs(fftconvolve(feature, wavelet, mode='same')))
                features.append(new_features)

            #add subsampled image to features
            start_index = end_index
            end_index += subsampled_image_size
            scattering_features[i, start_index:end_index] = average_and_subsample(image, image_subsample_size).ravel()

            #subsample features
            for k in range(1, len(features)):
                features_scale_n = features[k]
                subsample_size = subsample_sizes[k-1]
                feature_scale_n_size = nbpixels / subsample_size**2

                for feature_scale_n in features_scale_n:
                    start_index = end_index
                    end_index += feature_scale_n_size
                    scattering_features[i, start_index:end_index] = average_and_subsample(feature_scale_n, subsample_size).ravel()

    if normalize_features:
        return normalize(scattering_features)
    else:
        return scattering_features


def linear_kernel(features):
    return np.dot(features.T,features)


def distance_matrix(x, y):
    """Return the distance matrix ||x-y||.

    :param x: a set of p points in dimension d (p*d)
    :param y: a set of n points in dimension d (n*d)
    :returns: a p*n 2d array of distances
    """
    gram_matrix = np.dot(x,y.T)
    norms_x = np.sum(np.power(np.absolute(x),2),axis=1,keepdims=True)
    norms_y = np.sum(np.power(np.absolute(y),2), axis=1,keepdims=True)
    dist_matrix = norms_x + norms_y.T - 2 * gram_matrix
    dist_matrix = np.sqrt(np.maximum(dist_matrix,0))
    return dist_matrix


def gaussian(z,sigma):
    return np.exp(-z**2/2/sigma**2)

def gaussian_kernel(features,sigma):
    return gaussian(distance_matrix(features, features),sigma)

def cauchy(z,sigma):
    return 1 / (1 + z ** 2 / sigma ** 2)

def cauchy_kernel(features, sigma):
    return cauchy(distance_matrix(features, features),sigma)

