import numpy as np
import pywt
import pywt.data
from utils import *
from scipy.signal import fftconvolve

def wavelet_transform(Xtr):
    result = np.zeros((Xtr.shape[0], 972))

    for i in range(Xtr.shape[0]):
        r = Xtr[i][:1024].reshape(32,32)
        g = Xtr[i][1024:2048].reshape(32,32)
        b = Xtr[i][-1024:].reshape(32,32)

        rgbArray = np.zeros((32,32,3), 'uint8')
        rgbArray[..., 0] = (r+r.min())/(r.max()-r.min())*256
        rgbArray[..., 1] = (g+g.min())/(g.max()-g.min())*256
        rgbArray[..., 2] = (b+b.min())/(b.max()-b.min())*256

        grayImage = 0.2989 *rgbArray[..., 0]+ 0.5870 *rgbArray[..., 1]+0.1140 *rgbArray[..., 2]
        coeffs2 = pywt.dwt2(grayImage, 'bior1.3')
        LL, (LH, HL, HH) = coeffs2

        result[i,:] = np.concatenate((LH.ravel(), HL.ravel(), HH.ravel()), axis=0)
    return result

def fourier_modulus_1D_kernel(images, signal='rows'):
    """
        simply returns the fourier transform of the rows/columns of the image
    """

    fourier_1D_features_rows = np.zeros_like(images)
    fourier_1D_features_columns = np.zeros_like(images)

    for i in range(images.shape[0]):
        RGB = separate_RGB_images(images[i])

        for j, image in enumerate(RGB):
            rows = image
            columns = image.reshape((32,32)).T.reshape(1024)

            fourier_1D_features_rows[i,j*1024:(j+1)*1024] = np.abs(np.fft.fft(rows))
            fourier_1D_features_columns[i,j*1024:(j+1)*1024] = np.abs(np.fft.fft(columns))

    if signal == 'rows':
        return fourier_1D_features_rows
    elif signal == 'columns':
        return fourier_1D_features_columns
    else:
        return np.concatenate([fourier_1D_features_rows, fourier_1D_features_columns])

def fourier_modulus_1D_kernel_2(images):
    """
        simply returns the fourier transform of the image seen as a 1D array
    """
    n_images = images.shape[0]
    fourier_1D_features = np.zeros((n_images, 2*3*1024))

    for i in range(n_images):
        RGB = separate_RGB_images(images[i])

        for j, image in enumerate(RGB):
            fourier_1D_features[i, (2*j)*1024: (2*j+1)*1024] = np.abs(np.fft.fft(image))
            fourier_1D_features[i, (2*j+1)*1024: (2*j+2)*1024] = np.abs(np.fft.fft(image.reshape((32,32)).T.reshape(1024)))

    return fourier_1D_features

def fourier_modulus_2D_kernel(images):
    """
        returns 2Dthe fourier transform of the image seen
    """
    fourier_2D_modulus = np.zeros_like(images)

    for i in range(images.shape[0]):
        RGB = separate_RGB_images(images[i])

        for j, image in enumerate(RGB):
            fourier_2D_modulus[i, j*1024: (j+1)*1024] = np.abs(np.fft.fft2(image)).reshape(1024)

    return fourier_2D_modulus

def fourier_phase_2D_kernel(images):
    """
        returns the fourier transform of the image seen
    """
    fourier_2D_phase = np.zeros_like(images)

    for i in range(images.shape[0]):
        RGB = separate_RGB_images(images[i])

        for j, image in enumerate(RGB):
            fourier_2D_phase[i, j*1024: (j+1)*1024] = np.angle(np.fft.fft2(image)).reshape(1024)

    return fourier_2D_phase

def scattering_kernel(images, order, scale, wavelet_type='gabor'):
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

    subsampled_image_size = 1024 / image_subsample_size**2

    # compute size of scattering transform
    feature_size = subsampled_image_size
    n_feature_maps = 1
    for i, (scale, subsample_size) in enumerate(zip(scales, subsample_sizes)):
        wavelets = scale_wavelets[i]
        n_feature_maps *= len(wavelets)
        feature_size += n_feature_maps * 1024 / subsample_size**2

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
                feature_scale_n_size = 1024 / subsample_size**2

                for feature_scale_n in features_scale_n:
                    start_index = end_index
                    end_index += feature_scale_n_size
                    scattering_features[i, start_index:end_index] = average_and_subsample(feature_scale_n, subsample_size).ravel()

    if normalize_features:
        return normalize(scattering_features)
    else:
        return scattering_features
