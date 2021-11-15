from scipy.io import wavfile
import numpy as np
from skimage.restoration import denoise_wavelet
import matplotlib.pyplot as plt
import sounddevice as sd

Fs, x = wavfile.read('sample.wav') #Reading Audio Wave File
x = x/max(x) #Normalizing Amplitude

sigma = 0.05 #Niose Variance
x_noisy = x + sigma * np.random.randn(x.size) #Adding Noise to Signal

#Wavelet Denoising
x_denoise = denoise_wavelet(x_noisy, method = 'VisuShrink', mode = 'soft', wavelet_levels = 3, wavelet = 'sym8', rescale_sigma = 'True')
sd.play(x_noisy,Fs)
sd.play(x_denoise, Fs)

plt.figure(figsize=(20,10), dpi=100)
plt.plot(x_noisy)
plt.plot(x_denoise)
plt.show()