import matplotlib.pyplot as plt
from lab1_solution import *

if __name__ == "__main__":
    # Пример проверки для частоты 440 Гц (нота Ля)
    f_test = 440
    fs_test = 44100
    t_test = 0.01 # (10 мс)

    # Генерируем сигнал
    x = tone(f=f_test, t=t_test, waveform='harmonic', fs=fs_test)
    t_axis = np.linspace(0, t_test, len(x))

    plt.figure(figsize=(10, 4))
    plt.stem(t_axis, x) 
    plt.title(f"Визуальная проверка функции tone (f={f_test} Гц)")
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда")
    plt.grid(True)
    plt.show()

    from scipy.io.wavfile import write

    # Генерируем сигнал длительностью 2 секунды для прослушивания
    f_listen = 329.628
    duration = 2.0
    x_audio = tone(f_listen, duration, waveform='harmonic')

    write('tone_check.wav', 44100, (x_audio * 32767).astype(np.int16))