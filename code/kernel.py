import numpy as np

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
