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

def scattering_with_haar_wavelets(images):
    wavelet_1_hor = np.zeros((32,32))
    wavelet_1_vert = np.zeros((32,32))
    wavelet_1_hor[0:4,0:2] = 1.0 / 4**2
    wavelet_1_hor[0:4,2:4] = -1.0 / 4**2
    wavelet_1_vert[0:2,0:4] = 1.0 / 4**2
    wavelet_1_vert[2:4,0:4] = -1.0 / 4**2


    wavelet_2_hor = np.zeros((32,32))
    wavelet_2_vert = np.zeros((32,32))
    wavelet_2_hor[0:8,0:4] = 1.0 / 4**2
    wavelet_2_hor[0:4,4:8] = -1.0 / 4**2
    wavelet_2_vert[0:4,0:8] = 1.0 / 4**2
    wavelet_2_vert[4:8,0:8] = -1.0 / 4**2

    scattering_features = np.zeros((images.shape[0], 3 * 6 * 1024))
    for i in range(images.shape[0]):
        RGB = separate_RGB_images(images[i])

        for j, image in enumerate(RGB):
            features = []
            image = image.reshape((32, 32))
            image_w1h = np.abs(convolution_2D(image, wavelet_1_hor))
            image_w1v = np.abs(convolution_2D(image, wavelet_1_vert))
            features.append(image_w1h, image_w1v)
            features.append(np.abs(convolution_2D(image_w1h, wavelet_2_hor)))
            features.append(np.abs(convolution_2D(image_w1h, wavelet_2_vert)))
            features.append(np.abs(convolution_2D(image_w1v, wavelet_2_hor)))
            features.append(np.abs(convolution_2D(image_w1v, wavelet_2_vert)))

            for k, feat in enumerate(features):
                scattering_features[i,(6*j + k)*1024 :(6*j + k+1)*1024] = feat.reshape(1024)

    return scattering_features
