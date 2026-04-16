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
    # 1. Создаем временную сетку для выходного сигнала
    n_samples = len(x)
    t = np.arange(n_samples) / fs
    
    # 2. Вычисляем переменную задержку
    delay_t = dt + at * np.sin(2 * np.pi * f * t)
    
    # 3. Определяем моменты времени, из которых нужно "забрать" значения (t_source)
    t_source = t - delay_t
    
    # 4. Проверка на строгое возрастание отсчётов (избежание инверсии времени)
    if not np.all(np.diff(t_source) > 0):
        print("Warning: последовательный порядок отсчётов нарушен! "
              "Рекомендуется уменьшить параметры at или f.")
    
    # 5. Линейная интерполяция для определения значений в новых точках
    x_shifted = np.interp(t_source, t, x, left=0, right=0)
    
    return x_shifted

def apply_chorus(input_file, output_file):
    fs, data = wavfile.read(input_file)
    
    if len(data.shape) > 1:
        x = data[:, 0].astype(np.float32)
    else:
        x = data.astype(np.float32)
    
    x /= np.max(np.abs(x))
    
    shift1 = shift(x, fs, dt=0.020, at=0.010, f=3.0)
    shift2 = shift(x, fs, dt=0.025, at=0.008, f=2.5)
    
    y = x + shift1 + shift2
    
    y /= np.max(np.abs(y))
    
    wavfile.write(output_file, fs, (y * 32767).astype(np.int16))
    print(f"Эффект наложен. Файл сохранен как: {output_file}")

if __name__ == "__main__":
    apply_chorus('input.wav', 'output_chorus.wav')