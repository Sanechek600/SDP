import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.signal.windows import triang
import statsmodels.tsa.stattools as sm
import pyreaper


def my_acf(x: np.ndarray, m: int) -> float:
    """
    Вычисляет значение автокорреляционной функции для заданного сдвига m.
    Формула: R[m] = 1/(N-m) * sum_{k=0}^{N-m-1} (x[k] - mu) * (x[k+m] - mu)
    """
    N = len(x)
    if m < 0 or m >= N:
        raise ValueError("m должно быть в диапазоне [0, N-1]")
    mu = np.mean(x)
    # Используем срезы для эффективного вычисления
    x1 = x[:N-m] - mu
    x2 = x[m:] - mu
    acf_m = np.dot(x1, x2) / (N - m)

    x1 = x[:N] - mu
    x2 = x - mu
    acf_0 = np.dot(x1, x2) / N

    return acf_m / acf_0 if acf_0 != 0 else 0.0


def my_dtft(x: np.ndarray, fs: int, f: np.ndarray) -> np.ndarray:
    """
    Вычисляет амплитудный спектр сигнала x на частотах f (в Гц) с помощью ДВПФ.
    Если f - массив, возвращает массив амплитуд.
    Реализация через скалярное произведение (векторизована для одного f).
    """
    N = len(x)
    n = np.arange(N)
    # Обработка скалярного и векторного случая
    if np.isscalar(f):
        omega = 2 * np.pi * f / fs
        exp_part = np.exp(-1j * omega * n)
        X = np.dot(x, exp_part)
        return np.abs(X)
    else:
        # Векторный случай: вычисляем для каждого f
        X_abs = np.zeros_like(f, dtype=float)
        for i, freq in enumerate(f):
            omega = 2 * np.pi * freq / fs
            exp_part = np.exp(-1j * omega * n)
            X_abs[i] = np.abs(np.dot(x, exp_part))
        return X_abs


def psola(x: np.ndarray, fs: int, k: float) -> np.ndarray:
    """
    Изменяет частоту основного тона речи на основе алгоритма PSOLA.

    Parameters:
        x (np.ndarray): Входной одноканальный сигнал.
        fs (int): Частота дискретизации.
        k (float): Коэффициент изменения частоты.

    Returns:
        np.ndarray: Синтезированный сигнал с измененным основным тоном.
    """
    # Алгоритм Overlap-Add:
    # y(t) = sum_i w_i(t - t'_i) * x(t - t_i)
    
    x_norm = x.astype(np.float32)
    if np.max(np.abs(x_norm)) > 0:
        x_norm = x_norm / np.max(np.abs(x_norm))
    
    x_int16 = (x_norm * np.iinfo(np.int16).max).astype(np.int16)
    
    pm_times, pm, _, _, _ = pyreaper.reaper(x_int16, fs)
    
    pitch_marks = pm_times[pm == 1]
    
    if len(pitch_marks) < 2:
        return x.copy()
    
    pm_idx = (pitch_marks * fs).astype(int)

    y = np.zeros(int(len(x) * max(2.0, k)) + fs)
    new_center = 0.0
    max_period = int(fs / 50)  # Порог (50 Гц) для отделения голоса от шума/пауз

    for i in range(1, len(pm_idx)):
        T = pm_idx[i] - pm_idx[i-1]
        
        # глухие или паузы
        if T > max_period:
            unvoiced_segment = x[pm_idx[i-1] : pm_idx[i]].astype(np.float64)
            
            target_start = int(new_center)
            target_end = target_start + len(unvoiced_segment)
            
            if target_end < len(y):
                y[target_start:target_end] = unvoiced_segment
            
            new_center += len(unvoiced_segment)
            
        # гласные
        else:
            # L = 2 * T_0
            start_idx = pm_idx[i] - T
            end_idx = pm_idx[i] + T
            
            if start_idx >= 0 and end_idx < len(x):
                segment = x[start_idx:end_idx].astype(np.float64).copy()
                
                window = triang(len(segment))
                segment *= window
                
                target_start = int(new_center)
                target_end = target_start + len(segment)
                
                if target_end < len(y):
                    y[target_start:target_end] += segment
                
                # t'_i = t'_{i-1} + k * T_0
                new_center += k * T

    return y[:int(new_center + fs * 0.1)]


if __name__ == "__main__":
    # ---- 1. Загрузка сигнала ----
    filename = "speech.wav" 
    fs, x = wavfile.read(filename)
    # Если стерео, берём первый канал
    if x.ndim > 1:
        x = x[:, 0]
    # Приводим к float32 с диапазоном [-1, 1]
    x = x.astype(np.float32) / np.max(np.abs(x))

    # Для ускорения тестов и анализа возьмём фрагмент длительностью ~3 секунды с речью
    start_sec = 0.5
    end_sec = 3.5
    start_idx = int(start_sec * fs)
    end_idx = int(end_sec * fs)
    if end_idx > len(x):
        end_idx = len(x)
    x_seg = x[start_idx:end_idx].copy()
    fs_seg = fs  # частота дискретизации не меняется

    # ---- 2. Проверка my_acf на маленьком сегменте ----
    test_len = 20
    x_test = x_seg[:test_len]
    # Вычисляем библиотечную АКФ (adjusted=True)
    acf_lib = sm.acf(x_test, adjusted=True, nlags=test_len-1)
    # Сравниваем для нескольких m
    print("Проверка my_acf (первые 5 значений):")
    for m in range(5):
        my_val = my_acf(x_test, m)

        # Библиотечная
        lib_val = acf_lib[m]
        print(f"m={m}: my_acf={my_val:.6f}, statsmodels={lib_val:.6f}, разница={abs(my_val-lib_val):.2e}")

    # ---- 3. Оценка основного тона по АКФ ----
    # Для оценки используем сегмент x_seg (более продолжительный)
    # Вычисляем АКФ для всех m (библиотечной функцией, быстрее)
    acf_full = sm.acf(x_seg, adjusted=True, nlags=len(x_seg)-1)
    m_axis = np.arange(len(acf_full))

    # Поиск первого значимого пика в диапазоне частот 70–400 Гц
    min_f = 70
    max_f = 400
    min_period = int(np.round(fs_seg / max_f))  # максимальный период в отсчётах
    max_period = int(np.round(fs_seg / min_f))  # минимальный период
    # Ищем пик в интервале [min_period, max_period] (исключая m=0)
    search_region = acf_full[min_period:max_period+1]
    peak_idx_local = np.argmax(search_region)
    peak_m = min_period + peak_idx_local
    f0_acf = fs_seg / peak_m

    # График АКФ
    plt.figure(figsize=(10, 4))
    plt.plot(m_axis, acf_full)
    plt.axvline(float(peak_m), color='red', linestyle='--', label=f'Оценка F0 = {f0_acf:.1f} Гц')
    plt.xlim(0, max_period*2)
    plt.xlabel('Задержка m (отсчёты)')
    plt.ylabel('АКФ')
    plt.title('Автокорреляционная функция')
    plt.legend()
    plt.grid(True)
    plt.savefig('acf_plot.png')
    plt.show()

    # ---- 4. Оценка основного тона по ДВПФ ----
    freq_range = np.arange(min_f, max_f+1, 1.0)  # шаг 1 Гц
    spectrum = my_dtft(x_seg, fs_seg, freq_range)
    # Поиск максимума спектра
    peak_idx_fft = np.argmax(spectrum)
    f0_dtft = freq_range[peak_idx_fft]

    plt.figure(figsize=(10, 4))
    plt.plot(freq_range, spectrum)
    plt.axvline(f0_dtft, color='red', linestyle='--', label=f'Оценка F0 = {f0_dtft:.1f} Гц')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитудный спектр')
    plt.title('ДВПФ (амплитудный спектр)')
    plt.legend()
    plt.grid(True)
    plt.savefig('dtft_plot.png')
    plt.show()

    # ---- 5. Оценка с помощью REAPER ----
    # Подготовка данных для REAPER (int16)
    int16_info = np.iinfo(np.int16)
    x_norm = x / np.max(np.abs(x))
    x_int16 = (x_norm * min(int16_info.min, int16_info.max)).astype(np.int16)

    pm_times, pm, f_times, f, _ = pyreaper.reaper(x_int16, fs)
    # Усредняем частоту на вокализованных участках (f != -1)
    f_valid = f[f != -1]
    if len(f_valid) > 0:
        f0_reaper = np.mean(f_valid)
    else:
        f0_reaper = np.nan

    # Визуализация результатов REAPER
    plt.figure(figsize=(10, 6))
    t = np.arange(len(x)) / fs
    plt.subplot(2,1,1)
    plt.plot(t, x)
    plt.scatter(pm_times[pm == 1], x[(pm_times[pm == 1]*fs).astype(int)], marker='x', color='red', s=30)
    plt.title('Речевой сигнал и pitch marks')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')

    plt.subplot(2,1,2)
    plt.plot(f_times, f, '.-')
    plt.axhline(f0_reaper, color='red', linestyle='--', label=f'Среднее = {f0_reaper:.1f} Гц')
    plt.xlabel('Время (с)')
    plt.ylabel('Частота (Гц)')
    plt.title('Частота основного тона (REAPER)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reaper_plot.png')
    plt.show()

    # ---- Таблица оценок ----
    print("\n=== Таблица оценок частоты основного тона (Гц) ===")
    print(f"Метод АКФ:       {f0_acf:.2f}")
    print(f"Метод ДВПФ:      {f0_dtft:.2f}")
    print(f"Google REAPER:   {f0_reaper:.2f}")

    # Сохраняем таблицу в текстовый файл
    with open("estimates.txt", "w") as f_tab:
        f_tab.write("Оценки частоты основного тона (Гц):\n")
        f_tab.write(f"АКФ: {f0_acf:.2f}\n")
        f_tab.write(f"ДВПФ: {f0_dtft:.2f}\n")
        f_tab.write(f"REAPER: {f0_reaper:.2f}\n")

    # ---- 6. Применение PSOLA и сохранение результата ----
    try:
        k = 0.75  # коэффициент повышения частоты (можно изменить)
        y_psola = psola(x, fs, k)
        # Нормализуем выходной сигнал
        y_psola = y_psola / np.max(np.abs(y_psola))
        # Сохраняем как int16
        y_int16 = (y_psola * min(int16_info.min, int16_info.max)).astype(np.int16)
        wavfile.write("output_psola.wav", fs, y_int16)
        print(f"\nСигнал с изменённым тоном (k={k}) сохранён в output_psola.wav")
    except Exception as e:
        print(f"Ошибка в PSOLA: {e}")

    print("\nСкрипт выполнен.")