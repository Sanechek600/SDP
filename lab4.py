# pylint: disable=invalid-name,missing-docstring,too-few-public-methods

import unittest

import unittest
import numpy as np
from scipy.fft import fft
from scipy.signal import stft
from scipy.io import wavfile
import matplotlib.pyplot as plt


def dft(x: np.ndarray) -> np.ndarray:
    N = len(x)                     
    n = np.arange(N)               
    m = n.reshape(-1, 1)           
    # Матрица поворачивающих множителей W = exp(-2πj * m * n / N)
    W = np.exp(-2j * np.pi * m * n / N)
    # Умножение матрицы на вектор (ДПФ)
    return W @ x

    return x


def real_stft(x: np.ndarray, segment: int, overlap: int) -> np.ndarray:
    n_samples = x.shape[0]
    step = segment - overlap
    
    if n_samples < segment:
        return np.zeros((segment // 2 + 1, 0))
        
    num_segments = (n_samples - overlap) // step
    num_bins = segment // 2 + 1
    
    result = np.zeros((num_bins, num_segments), dtype=np.complex128)
    
    for i in range(num_segments):
        start = i * step
        seg = x[start : start + segment]
        X = dft(seg)
        result[:, i] = X[:num_bins]
        
    return result


class Test(unittest.TestCase):
    class Params:
        def __init__(self, n: int, segment: int, overlap: int) -> None:
            self.n = n
            self.segment = segment
            self.overlap = overlap

        def __str__(self) -> str:
            return f"n={self.n} segment={self.segment} overlap={self.overlap}"

    def test_dft(self) -> None:
        for n in (10, 11, 12, 13, 14, 15, 16):
            with self.subTest(n=n):
                np.random.seed(0)
                x = np.random.rand(n) + 1j * np.random.rand(n)
                actual = dft(x)
                expected = fft(x)
                self.assertTrue(np.allclose(actual, expected))

    #@unittest.skip
    def test_stft(self) -> None:
        params_list = (
            Test.Params(50, 10, 5),
            Test.Params(50, 10, 6),
            Test.Params(50, 10, 7),
            Test.Params(50, 10, 8),
            Test.Params(50, 10, 9),
            Test.Params(101, 15, 7),
            Test.Params(101, 15, 8),
        )

        for params in params_list:
            with self.subTest(params=str(params)):
                np.random.seed(0)
                x = np.random.rand(params.n)
                actual = real_stft(x, params.segment, params.overlap)
                _, _, expected = stft(
                    x,
                    boundary=None,
                    nperseg=params.segment,
                    noverlap=params.overlap,
                    padded=False,
                    window="boxcar",
                )
                assert isinstance(expected, np.ndarray)
                self.assertTrue(np.allclose(actual, params.segment * expected))


def main() -> None:
    #unittest.main()

    fs, x = wavfile.read("6412-28.wav")
    if len(x.shape) > 1:
        x = x.mean(axis=1)

    f, t, spectrum = stft(x, fs, nperseg=4096, noverlap=2048)
    mag_squared = np.abs(spectrum) ** 2

    plt.figure('Spectrogram')
    plt.pcolormesh(t, f, mag_squared, shading='gouraud')
    plt.ylim(300, 1000)
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.title('STFT Spectrogram')

    plt.show()


if __name__ == "__main__":
    main()
