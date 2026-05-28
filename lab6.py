import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.stattools import acf
from scipy.signal import deconvolve, convolve, correlate
from scipy.fft import fft
from statsmodels.stats.diagnostic import acorr_ljungbox

from morse_encode import MORSE_CODES, morse_encode


def ensure_plots_dir():
    """Создаёт папку plots, если её нет"""
    if not os.path.exists("plots"):
        os.makedirs("plots")


def read_data(path):
    data = np.load(path)
    y = np.ravel(data[0, :])   
    v = np.ravel(data[1, :])     
    h = data[2:, :]            
    return y, v, h


def estimate_ir(h_all):
    h_est = np.mean(h_all, axis=0)[:200]
    h_est[np.abs(h_est) < 1e-4] = 0

    plt.figure(figsize=(10, 4))
    plt.stem(h_est)
    plt.title("Оценка импульсной характеристики h[n]")
    plt.grid(True)
    plt.savefig("plots/h_est.png", dpi=150)
    plt.show()
    return h_est


def noise_acf(v):
    r = acf(v, nlags=100)

    plt.figure(figsize=(10, 4))
    plt.stem(r)
    plt.title("Автокорреляция шума v[n]")
    plt.grid(True)
    plt.savefig("plots/noise_acf.png", dpi=150)
    plt.show()


def noise_spectrum(v):
    v_chunk = v[1000:3000]
    V = fft(v_chunk)
    V_power = np.abs(V)**2
    V_norm = V_power / np.mean(V_power)

    plt.figure(figsize=(10, 4))
    plt.plot(V_norm)
    plt.title("Спектр шума v[n]")
    plt.grid(True)
    plt.savefig("plots/noise_spectrum.png", dpi=150)
    plt.show()


def build_lowpass_filter(w0, n):
    mid = n // 2
    time_axis = np.arange(-mid, mid + 1)

    with np.errstate(divide='ignore', invalid='ignore'):
        h_lp = np.sin(w0 * time_axis) / (np.pi * time_axis)

    h_lp[mid] = w0 / np.pi
    h_lp *= np.hamming(len(h_lp))
    h_lp /= np.sum(h_lp)

    plt.figure(figsize=(8, 3))
    plt.plot(h_lp)
    plt.title(f"ИХ НЧ-фильтра w0 = {w0:.4f}")
    plt.grid(True)
    plt.savefig("plots/lowpass_filter_ir.png", dpi=150)
    plt.show()
    return h_lp


def calculate_M_w0(y, h_est):
    x_1, _ = deconvolve(y, h_est)
    N = len(x_1)
    spectrum = fft(x_1 - np.mean(x_1))
    
    half_len = N // 2
    spectrum_amp = np.abs(spectrum[:half_len])
    k = np.argmax(spectrum_amp)

    w0 = 2 * np.pi * k / N
    M = int(np.round(N / (2 * k)))

    plt.figure(figsize=(10, 4))
    plt.plot(np.abs(spectrum[:half_len]))
    plt.axvline(k, color='r', linestyle='--', label='w0')
    plt.axvline(k * 2, color='r', linestyle=':', label='2w0')
    plt.axvline(k * 3, color='r', linestyle='--', label='3w0')
    plt.title("Амплитудный спектр исходного сигнала")
    plt.legend()
    plt.savefig("plots/spectrum_original.png", dpi=150)
    plt.show()

    print(f"Частота среза w0 = {w0:.5f}")
    print(f"Размер точки M = {M}")
    return w0, M


def recover_signal(y, h_est, w0, M):
    h_n = build_lowpass_filter(w0, 51)
    y_filtered = convolve(y, h_n)
    x_recovered, _ = deconvolve(y_filtered, h_est)

    x_bin = (x_recovered > 0.5).astype(int)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(y_filtered, color='red')
    plt.title(f"После НЧ-фильтра (M={M})")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(x_recovered, label="восстановленный", color='orange')
    plt.step(range(len(x_bin)), x_bin, where='post', label="бинарный", color='blue')
    plt.title("Восстановленный сигнал")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/recovered_signal.png", dpi=150)
    plt.show()

    return x_recovered, x_bin


def segmented_view(x_bin):
    segments = []
    if len(x_bin) == 0:
        return []
    current = x_bin[0]
    length = 1
    for val in x_bin[1:]:
        if val == current:
            length += 1
        else:
            segments.append((current, length))
            current = val
            length = 1
    segments.append((current, length))
    return segments


def segmented_to_morse_code(segments, M):
    morse_code = ""
    for (val, length) in segments:
        if val == 1:                 
            if length < 1.5 * M:        
                morse_code += "."
            else:                   
                morse_code += "-"
        else:                          
            if length < 2 * M:         
                pass
            elif length < 4 * M:       
                morse_code += " "
            else:                      
                morse_code += "   "
    morse_code = morse_code.strip()
    return morse_code


def morse_code_to_text(morse_code):
    decode = {v: k for k, v in MORSE_CODES.items()}
    words = morse_code.split("   ")
    result = []
    for word in words:
        letters = word.split(" ")
        decoded_letters = []
        for letter in letters:
            if letter in decode:
                decoded_letters.append(decode[letter])
        result.append("".join(decoded_letters))
    return " ".join(result)


def decode_message(x_bin, M):
    runs = segmented_view(x_bin)
    morse_code = segmented_to_morse_code(runs, M)
    text = morse_code_to_text(morse_code)
    print("Сигнал в форме кода Морзе:", morse_code)
    print("Расшифрованный текст:", text)
    return text


def calculate_mse(recovered, decoded_text, M):
    ideal = morse_encode(decoded_text, M)
    corr_idx = correlate(recovered, ideal, mode='full')
    delay = np.argmax(corr_idx) - len(ideal) + 1
    recovered_aligned = np.roll(recovered, -delay)[:len(ideal)]
    mse = np.mean((ideal - recovered_aligned) ** 2)
    print(f"СКО: {mse:.6f}")
    return mse


def main():
    ensure_plots_dir()  # создаём папку для графиков

    y, v, h_all = read_data("6412-26.npy")

    # Анализ шума
    lb = acorr_ljungbox(v, lags=[10, 30, 50], return_df=True)
    print("\nQ-тест Льюнг-Бокса:")
    print(lb)
    
    h_est = estimate_ir(h_all)
    noise_acf(v)
    noise_spectrum(v)

    max_val = np.max(np.abs(h_est))
    significant_indices = np.where(np.abs(h_est) > 0.05 * max_val)[0]

    print("\nЗначимые индексы:", significant_indices)
    print("Их значения для формулы:")
    for idx in significant_indices:
        print(f"h[{idx}] = {h_est[idx]:.6f}")
    
    w0, M = calculate_M_w0(y, h_est)
    x_recovered, x_bin = recover_signal(y, h_est, w0, M)
    decoded_text = decode_message(x_bin, M)
    calculate_mse(x_recovered, decoded_text, M)


if __name__ == "__main__":
    main()