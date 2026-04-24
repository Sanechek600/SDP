import unittest

import numpy as np
from scipy.signal import stft, istft
from scipy.fft import ifft
import scipy.io.wavfile as wavfile


def idft(x: np.ndarray) -> np.ndarray:
    N = len(x)
    k = np.arange(N)
    m = k.reshape((N, 1))
    W = np.exp(1j * 2 * np.pi * k * m / N)
    return (W @ x) / N


def real_istft(spectrum: np.ndarray, segment: int, overlap: int) -> np.ndarray:
    assert len(spectrum.shape) == 2
    assert spectrum.shape[0] == segment // 2 + 1

    hop = segment - overlap
    num_frames = spectrum.shape[1]
    
    out_len = (num_frames - 1) * hop + segment
    x = np.zeros(out_len)
    w_sum = np.zeros(out_len)
    
    for t in range(num_frames):
        X_one = spectrum[:, t]
        if segment % 2 == 0:
            X_full = np.concatenate([X_one, np.conj(X_one[-2:0:-1])])
        else:
            X_full = np.concatenate([X_one, np.conj(X_one[-1:0:-1])])
        
        y_t = np.real(idft(X_full))
        
        start = t * hop
        x[start:start+segment] += y_t
        w_sum[start:start+segment] += 1.0
        
    nonzero = w_sum > 0
    x[nonzero] /= w_sum[nonzero]
    
    return x


class Test(unittest.TestCase):
    class Params:
        def __init__(self, n: int, segment: int, overlap: int) -> None:
            self.n = n
            self.segment = segment
            self.overlap = overlap

        def __str__(self) -> str:
            return f"n={self.n} segment={self.segment} overlap={self.overlap}"

    def test_idft(self) -> None:
        for n in (10, 11, 12, 13, 14, 15, 16):
            with self.subTest(n=n):
                np.random.seed(0)
                x = np.random.rand(n) + 1j * np.random.rand(n)
                actual = idft(x)
                expected = ifft(x)
                self.assertTrue(np.allclose(actual, expected))

    def test_istft_unmodified(self) -> None:
        self._test_istft(False)

    def test_istft_modified(self) -> None:
        self._test_istft(True)

    def _test_istft(self, modify: bool) -> None:
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

                _, _, s = stft(
                    x,
                    boundary=None,
                    nperseg=params.segment,
                    noverlap=params.overlap,
                    padded=False,
                    window="boxcar",
                )

                assert isinstance(s, np.ndarray)

                if modify:
                    low_pass_filter = np.concatenate(
                        (
                            np.ones(s.shape[0] // 2),
                            np.zeros(s.shape[0] - s.shape[0] // 2),
                        )
                    )
                    for column in np.arange(s.shape[1]):
                        s[:, column] = s[:, column] * low_pass_filter

                _, expected = istft(
                    s,
                    boundary=None,
                    nperseg=params.segment,
                    noverlap=params.overlap,
                    window="boxcar",
                )

                assert isinstance(expected, np.ndarray)

                actual = real_istft(s * params.segment, params.segment, params.overlap)

                self.assertTrue(np.allclose(actual, expected))


def main() -> None:
    unittest.main(exit=False)

    sample_rate, data = wavfile.read("./voice/input.wav")
    
    segment_ms = 20
    nperseg = int(sample_rate * segment_ms / 1000)
    noverlap = nperseg // 2
    window = 'hann'
    
    if data.ndim > 1:
        data_t = data.T
    else:
        data_t = data
        
    f, t, Zxx = stft(data_t, fs=sample_rate, window=window, nperseg=nperseg, noverlap=noverlap)
    
    Zxx_robot = np.abs(Zxx)
    
    _, data_robot_t = istft(Zxx_robot, fs=sample_rate, window=window, nperseg=nperseg, noverlap=noverlap)
    
    if data.ndim > 1:
        data_robot = data_robot_t.T
    else:
        data_robot = data_robot_t
        
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        data_robot = np.clip(data_robot, info.min, info.max)
    data_robot = data_robot.astype(data.dtype)
    
    wavfile.write("./voice/output.wav", sample_rate, data_robot)


if __name__ == "__main__":
    main()