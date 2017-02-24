import numpy as np
import pywt
import pywt.data
from utils import *

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

def scattering_transform(images, wavelet_type='gabor'):
    if wavelet_type == 'gabor':
        wavelets_scale_1 = generate_gabor_wavelets_2D(4,32)
        wavelets_scale_2 = generate_gabor_wavelets_2D(8,32)
        wavelets_scale_3 = generate_gabor_wavelets_2D(16,32)
    else:
        wavelets_scale_1 = generate_haar_wavelets_2D(4,32)
        wavelets_scale_2 = generate_haar_wavelets_2D(8,32)
        wavelets_scale_3 = generate_haar_wavelets_2D(16,32)

    scale_1_subsample_size = 2
    scale_2_subsample_size = 4
    scale_3_subsample_size = 8

    feature_size = len(wavelets_scale_1) * (1024 / scale_1_subsample_size**2) + len(wavelets_scale_2) * len(wavelets_scale_1) * (1024 / scale_2_subsample_size**2)

    scattering_features = np.zeros((images.shape[0], 3 * feature_size))

    for i in range(images.shape[0]):
        RGB = separate_RGB_images(images[i])

        for j, image in enumerate(RGB):
            image = image.reshape((32,32))
            features_scale_1 = []

            for wavelet_scale_1 in wavelets_scale_1:
                features_scale_1.append(np.abs(convolution_2D(image, wavelet_scale_1)))

            features_scale_2 = []

            for wavelet_scale_2 in wavelets_scale_2:
                for feature_scale_1 in features_scale_1:
                    features_scale_2.append(np.abs(convolution_2D(feature_scale_1, wavelet_scale_2)))

            size_1 = 1024 / scale_1_subsample_size**2
            size_2 = 1024 / scale_2_subsample_size**2

            end = j * feature_size

            for feature_scale_1 in features_scale_1:
                start = end
                end += size_1

                scattering_features[i, start:end] = average_and_subsample(feature_scale_1, scale_1_subsample_size).reshape(size_1)

            for feature_scale_2 in features_scale_2:
                start = end
                end += size_2

                scattering_features[i, start:end] = average_and_subsample(feature_scale_2, scale_2_subsample_size).reshape(size_2)

    return scattering_features
