import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal


def tone(f, t, waveform='harmonic', fs=44100):
    """
    3.1 Генерация чистого тона.
    Возвращает массив numpy.float с амплитудой в диапазоне [-1, 1].
    """
    n = np.arange(int(fs * t))
    t_axis = n / fs
    
    if waveform == 'harmonic':
        x = np.cos(2 * np.pi * f * t_axis)
    elif waveform == 'square':
        x = signal.square(2 * np.pi * f * t_axis, duty=0.5)
    elif waveform == 'triangle':
        x = signal.sawtooth(2 * np.pi * f * t_axis, width=0.5)
    elif waveform == 'sawtooth':
        x = signal.sawtooth(2 * np.pi * f * t_axis)
    else:
        raise ValueError("Неверный тип waveform. Используйте: harmonic, square, triangle или sawtooth.")
    
    return x.astype(np.float64)

def musical_tone(f, t, waveform='harmonic', fs=44100, db=-20):
    """
    3.2 Генерация затухающего составного тона.
    Складывает обертоны до 20кГц и применяет затухание.
    """
    # 1. Формирование составного сигнала (основной тон + гармоники)
    total_signal = np.zeros(int(fs * t))
    k = 1
    while f * k < 20000:
        # Амплитуда гармоники уменьшается с ее номером
        total_signal += (1/k) * tone(f * k, t, waveform, fs)
        k += 1
    
    # 2. Нормировка амплитуды к 1
    total_signal = total_signal / np.max(np.abs(total_signal))
    
    # 3. Применение затухания (экспонента a^n)
    # Коэффициент a вычисляется: a^(N-1) = 10^(db/20)
    N = len(total_signal)
    if db != 0:
        R = 10**(db / 20)
        a = R**(1 / (N - 1))
        envelope = a**np.arange(N)
        total_signal *= envelope
        
    return total_signal

if __name__ == "__main__":
    # --- 3.3 Творческая часть ---

    # Частоты нот первой и второй октавы (Гц)
    NOTES = {
        'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23, 'G4': 392.00, 'A4': 440.00, 'B4': 493.88,
        'C5': 523.25, 'D5': 587.33, 'E5': 659.25, 'F5': 698.46, 'G5': 783.99, 'A5': 880.00, 'B5': 987.77,
        'P': 0 # Пауза
    }

    # Мелодия: массив кортежей (нота, длительность в долях)
    melody_data = [
        ('E4', 1), ('D4', 1), ('D4', 1), ('E4', 1), ('D4', 2), ('G4', 2),
        ('E4', 1), ('E4', 1), ('E4', 2), ('E4', 1), ('E4', 1), ('E4', 2),
        ('E4', 1), ('G4', 1), ('C4', 1.5), ('D4', 0.5), ('E4', 4),
        ('F4', 1), ('F4', 1), ('F4', 1.5), ('F4', 0.5), ('F4', 1), 
        ('E4', 1), ('E4', 1), ('E4', 0.5), ('E4', 0.5)
    ]

    tempo = 0.32 # Базовая длительность доли (сек)
    fs = 44100
    composition = np.array([])

    print("Генерация мелодии...")
    for note_name, duration in melody_data:
        t_note = duration * tempo
        if note_name == 'P':
            chunk = np.zeros(int(fs * t_note))
        else:
            # Генерируем "музыкальный" затухающий тон
            chunk = musical_tone(NOTES[note_name], t_note, waveform='harmonic', db=-25)
        
        composition = np.concatenate((composition, chunk))

    total_duration = len(composition) / fs
    print(f"Итоговая длительность: {total_duration:.2f} сек.")

    # Сохранение в WAV
    wav_data = (composition * 32767).astype(np.int16)
    wavfile.write('lab1_melody.wav', fs, wav_data)

    # --- Визуализация ---
    plt.figure(figsize=(12, 6))

    # График 1: Форма одного составного тона (первые 20 мс)
    plt.subplot(2, 1, 1)
    sample_tone = musical_tone(NOTES['C4'], 0.1, waveform='triangle', db=-10)
    t_axis = np.linspace(0, 0.1, len(sample_tone))
    plt.plot(t_axis[:882], sample_tone[:882]) # первые ~20 мс при 44100 Гц
    plt.title("Осциллограмма составного тона (нота C4, треугольная форма, 20 мс)")
    plt.xlabel("Время (с)")
    plt.ylabel("Амплитуда")
    plt.grid()

    # График 2: Огибающая всей мелодии (затухание)
    plt.subplot(2, 1, 2)
    plt.plot(composition[::100]) # Прореживание для скорости отрисовки
    plt.title("Общая огибающая композиции (визуализация затухания нот)")
    plt.xlabel("Отсчеты (прорежено x100)")
    plt.ylabel("Амплитуда")
    plt.tight_layout()
    plt.show()

    print("Файл 'lab1_melody.wav' сохранен.")
