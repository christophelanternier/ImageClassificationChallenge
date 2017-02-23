import numpy as np
import pywt
import pywt.data

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

def fourier_modulus_1D_kernel(images):
    """
        simply returns the fourier transform of the image seen as a 1D array
    """
    fourier_1D_features = np.zeros_like(images)

    for i in range(images.shape[0]):
        fourier_1D_features[i,:] = np.abs(np.fft.fft(images[i,:]))

    return fourier_1D_features

def fourier_modulus_2D_kernel(images):
    """
        returns 2Dthe fourier transform of the image seen
    """
    fourier_2D_modulus = np.zeros_like(images)

    for i in range(images.shape[0]):
        image_1 = images[i,0:1024].reshape((32, 32))
        image_2 = images[i,1024:2048].reshape((32, 32))
        image_3 = images[i,2048:3072].reshape((32, 32))

        fourier_2D_modulus[i,0:1024] = np.abs(np.fft.fft2(image_1)).reshape(1024)
        fourier_2D_modulus[i,1024:2048] = np.abs(np.fft.fft2(image_2)).reshape(1024)
        fourier_2D_modulus[i,2048:3072] = np.abs(np.fft.fft2(image_3)).reshape(1024)

    return fourier_2D_modulus

def fourier_phase_2D_kernel(images):
    """
        returns the fourier transform of the image seen
    """
    fourier_2D_phase = np.zeros_like(images)

    for i in range(images.shape[0]):
        image_1 = images[i,0:1024].reshape((32, 32))
        image_2 = images[i,1024:2048].reshape((32, 32))
        image_3 = images[i,2048:3072].reshape((32, 32))

        fourier_2D_phase[i,0:1024] = np.angle(np.fft.fft2(image_1)).reshape(1024)
        fourier_2D_phase[i,1024:2048] = np.angle(np.fft.fft2(image_2)).reshape(1024)
        fourier_2D_phase[i,2048:3072] = np.angle(np.fft.fft2(image_3)).reshape(1024)

    return fourier_2D_phase
