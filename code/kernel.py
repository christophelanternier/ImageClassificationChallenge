import numpy as np

def fourier_kernel(images):
    fourier_features = np.zeros_like(images)

    for i in range(images.shape[0]):
        fourier_features[i,:] = np.abs(np.fft.fft(images[i,:]))

    return fourier_features
