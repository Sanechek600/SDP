import matplotlib.pyplot as plt
from lab1.lab1_main import *
from scipy.io.wavfile import write


if __name__ == "__main__":
    # Check for A at 440 Hz
    f_test = 440
    fs_test = 44100
    t_test = 0.01 

    x = tone(f=f_test, t=t_test, waveform='harmonic', fs=fs_test)
    t_axis = np.linspace(0, t_test, len(x))

    plt.figure(figsize=(10, 4))
    plt.stem(t_axis, x) 
    plt.title(f"Visualized tone (f={f_test} Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.show()

    f_listen = 329.628
    duration = 2.0
    x_audio = tone(f_listen, duration, waveform='harmonic')

    write('tone_check.wav', 44100, (x_audio * 20000).astype(np.int16))