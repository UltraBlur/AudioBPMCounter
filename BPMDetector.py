import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
import os


def detectHeartBPM(input_file):
    try:
        sample_rate, audio_data = wavfile.read(input_file)
        print("Sample_rate:", sample_rate)

        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]

        # Design a bandpass filter
        filter_order = 3
        low_freq = 20
        high_freq = 800
        b, a = signal.butter(filter_order, [low_freq, high_freq], btype='band', fs=sample_rate)

        filtered_audio = signal.filtfilt(b, a, audio_data)

        output_wave_file(filtered_audio, output_path)
        
        # Use peak detection to find peaks of the heartbeat signal
        peaks, _ = signal.find_peaks(filtered_audio, distance=sample_rate*0.5)
        print("Peaks:", peaks)
        print("PeaksCount:", len(peaks))

        if len(peaks) > 1:  
            # Calculate BPM
            bpm = 60 / np.mean(np.diff(peaks)) * sample_rate

            # Print the BPM result
            print("Heartbeat BPM:", bpm)
        else:
            print("No peaks found in the audio.")

    except FileNotFoundError:
        print("File not found.")
    except ValueError:
        print("Invalid audio data.")
    except Exception as e:
        print("An error occurred:", str(e))
        
def detectBreathBPM(input_file):
    try:
        sample_rate, audio_data = wavfile.read(input_file)
        print("Sample_rate:", sample_rate)

        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]

        # Design a bandpass filter
        filter_order = 4
        low_freq = 400
        high_freq = 800
        b, a = signal.butter(filter_order, [low_freq, high_freq], btype='band', fs=sample_rate)

        filtered_audio = signal.filtfilt(b, a, audio_data)

        output_wave_file(filtered_audio, output_path2)
          
        # Use peak detection to find peaks of the heartbeat signal
        peaks, _ = signal.find_peaks(filtered_audio, distance=sample_rate*0.5)
        print("Peaks:", peaks)

        if len(peaks) > 1:
            # Calculate BPM
            bpm = 60 / np.mean(np.diff(peaks)) * sample_rate

            # Print the BPM result
            print("Breath BPM:", bpm)
        else:
            print("No peaks found in the audio.")

    except FileNotFoundError:
        print("File not found.")
    except ValueError:
        print("Invalid audio data.")
    except Exception as e:
        print("An error occurred:", str(e))

def output_wave_file(audio_data, output_path):

    export_dtype = np.int16  
    wavfile.write(output_path, 48000, audio_data.astype(export_dtype))
    
    


input_file = r'D:\Projects\Coding\OSC_BPMDetector\Audio\Heart_01.wav'
input_file2 = r'D:\Projects\Coding\OSC_BPMDetector\Audio\Breath_01.wav'
output_path = r'D:\Projects\Coding\OSC_BPMDetector\Audio\Audiofiltered_audio.wav'
output_path2 = r'D:\Projects\Coding\OSC_BPMDetector\Audio\Audiofiltered_Breath_audio.wav'

detectHeartBPM(input_file)
detectBreathBPM(input_file2)