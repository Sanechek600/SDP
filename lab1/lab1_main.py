import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal


def tone(f: float, t: float, waveform='harmonic', fs=44100):
    """
    Pure tone generator. Returns values in [-1, 1] 
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
        raise ValueError("Ineligible waveform. Use: harmonic, square, triangle, sawtooth.")
    
    return x.astype(np.float64)

def musical_tone(f: float, t: float, waveform='harmonic', fs=44100, db=-20):
    """
    Fading compound tone generator. Applies fade on sum of overtones up to 20kHz
    """
    # Compound tone
    end_signal = np.zeros(int(fs * t))
    k = 1
    while f * k < 20000:
        # Harmonics' amplitude fades with iteration
        end_signal += (1/k) * tone(f * k, t, waveform, fs)
        k += 1
    
    # Normalizing amplitude
    end_signal = end_signal / np.max(np.abs(end_signal))
    
    # Applying fade via a^n
    N = len(end_signal)
    if db != 0:
        a = (10**(db / 20))**(1 / (N - 1))
        envelope = a**np.arange(N)
        end_signal *= envelope
        
    return end_signal

if __name__ == "__main__":
    NOTES = {
        'C4': 261.63, 'D4': 293.66, 'E4': 329.63, 'F4': 349.23, 'G4': 392.00, 'A4': 440.00, 'B4': 493.88,
        'C5': 523.25, 'D5': 587.33, 'E5': 659.25, 'F5': 698.46, 'Fd5': 739.99, 'G5': 783.99, 'A5': 880.00, 'B5': 987.77,
        'C6': 1046.502, 
        'P': 0 
    }

    melody = [
        ('C5', 0.5), ('C5', 0.5), ('C5', 0.5), ('C5', 0.5), ('P', 2), 
        ('C5', 0.5), ('C5', 0.5), ('C5', 0.5), ('C5', 0.5), ('P', 2),
        ('C5', 0.5), ('C5', 0.5), ('C5', 0.5), ('E5', 0.5), ('P', 1),
        ('E5', 0.5), ('E5', 0.5), ('E5', 0.5), ('E5', 0.5), ('P', 1),
        ('E5', 0.5), ('E5', 0.5), ('E5', 0.5), ('G5', 0.5), ('P', 1),
        ('G5', 0.5), ('G5', 0.5), ('G5', 0.5), ('G5', 0.5), ('P', 1),
        ('G5', 0.5), ('C6', 2), ('B5', 2), ('Fd5', 0.5), ('A5', 1),
        ('G5', 1), ('F5', 1), ('D5', 1), ('C5', 1),
        ('B4', 0.5), ('C5', 0.5), ('D5', 1),
        ('G4', 0.5), ('D5', 0.5), ('E5', 1),
        ('C4', 0.5), ('E4', 0.5), ('G4', 0.5), ('C5', 0.5), ('E5', 0.5), ('G5', 0.5), ('C6', 2), 
    ]

    tempo = 0.32
    fs = 44100
    composition = np.array([])

    print("Generating melody...")
    for note_name, duration in melody:
        t_note = duration * tempo
        if note_name == 'P':
            chunk = np.zeros(int(fs * t_note))
        else:
            chunk = musical_tone(NOTES[note_name], t_note, waveform='harmonic', db=-25)
        
        composition = np.concatenate((composition, chunk))

    total_duration = len(composition) / fs
    print(f"Duration: {total_duration:.2f} seconds.")

    wav_data = (composition * 20000).astype(np.int16)
    wavfile.write('lab1_melody.wav', fs, wav_data)

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    sample_tone = musical_tone(NOTES['A5'], 0.1, waveform='sawtooth', db=-10)
    t_axis = np.linspace(0, 0.1, len(sample_tone))
    plt.plot(t_axis[:882], sample_tone[:882]) 
    plt.title("Compound tone oscillogram (A5, sawtooth, 20 ms)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()

    plt.subplot(2, 1, 2)
    plt.plot(composition[::100]) 
    plt.title("Composition envelope")
    plt.xlabel("Count (x100)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
