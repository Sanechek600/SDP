import numpy as np
from scipy.io import wavfile

def shift(x, fs, dt, at, f):
    """
    Функция для получения смещённой версии сигнала с переменной задержкой.
    x  - входной сигнал
    fs - частота дискретизации
    dt - постоянная часть задержки (с)
    at - амплитуда переменной части задержки (с)
    f  - частота изменения задержки (Гц)
    """
    n_samples = len(x)
    t = np.arange(n_samples) / fs
    
    delay_t = dt + at * np.sin(2 * np.pi * f * t)
    
    t_source = t - delay_t
    
    if not np.all(np.diff(t_source) > 0):
        print("Warning: последовательный порядок отсчётов нарушен! "
              "Рекомендуется уменьшить параметры at или f.")
    
    x_shifted = np.interp(t_source, t, x, left=0, right=0)
    
    return x_shifted

def apply_chorus(input_file, output_file):
    fs, data = wavfile.read(input_file)
    
    if len(data.shape) > 1:
        in_signal = data[:, 0].astype(np.float32)
    else:
        in_signal = data.astype(np.float32)
    
    in_signal /= np.max(np.abs(in_signal))
    
    shift1 = shift(in_signal, fs, dt=0.020, at=0.010, f=3.0)
    shift2 = shift(in_signal, fs, dt=0.025, at=0.008, f=2.5)
    
    out_signal = in_signal + shift1 + shift2
    
    out_signal /= np.max(np.abs(out_signal))
    
    wavfile.write(output_file, fs, (out_signal * 32767).astype(np.int16))
    print(f"Эффект наложен. Файл сохранен как: {output_file}")

if __name__ == "__main__":
    apply_chorus('input.wav', 'output_chorus.wav')