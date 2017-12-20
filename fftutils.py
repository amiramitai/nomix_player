import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wf
from scipy import signal

from pydub import AudioSegment
from pydub.utils import make_chunks


def windowed_fft(channel, fft_size=128, overlap_fac=0.5):
    """
    was taken from here:
    https://kevinsprojects.wordpress.com/2014/12/13/short-time-fourier-transform-using-python-and-numpy/
    """
    print('[+] time_to_freq', fft_size, channel.shape)
    hop_size = np.int32(np.floor(fft_size * overlap_fac))
    # The last segment can overlap the end of the data array by no more than one window size
    pad_end_size = fft_size
    total_segments = np.int32(np.ceil(len(channel) / np.float32(hop_size)))

    # our half cosine window
    # window = np.blackman(fft_size)
    window = signal.blackmanharris(fft_size)
    # the zeros which will be used to double each segment size
    inner_pad = np.zeros(fft_size)

    # the data to process
    proc = np.concatenate((channel, np.zeros(pad_end_size)))
    result = np.empty((total_segments, fft_size*2), dtype=np.complex)

    for i in range(total_segments):                       # for each segment
        current_hop = hop_size * i                        # figure out the current segment offset
        segment = proc[current_hop:current_hop+fft_size]  # get the current segment
        windowed = segment * window                       # multiply by the half cosine function
        padded = np.append(windowed, inner_pad)           # add 0s to double the length of the data
        spectrum = np.fft.fft(padded)                     # take the Fourier Transform
        spectrum = spectrum / fft_size                    # scale by the number of samples
        result[i, :] = spectrum               # append to the results array
    print('[+] out:', result.shape)
    return result


def windowed_ifft(bins, fft_size=128, overlap_fac=0.5, orig=None, dtype=np.int16):
    print('[+] freq_to_time', fft_size, bins.shape)
    hop_size = np.int32(np.floor(fft_size * overlap_fac))
    pad_end_size = fft_size
    total_segments = np.int32(np.ceil(len(bins) / np.float32(hop_size)))

    # window = np.blackman(fft_size)
    window = signal.blackmanharris(fft_size)
    res_size = int(len(bins) * fft_size * overlap_fac) + fft_size + 1
    result = np.empty(res_size, dtype=dtype)

    for i in range(len(bins)):
        scaled = bins[i] * fft_size
        padded = np.fft.ifft(scaled).real
        windowed = padded[:-fft_size]
        segment = windowed / window
        pcm = segment.astype(dtype)
        result[int(i*fft_size*overlap_fac):int(i*fft_size*overlap_fac)+fft_size] = pcm[:fft_size]

    print('[+] out:', result.shape)
    return result
