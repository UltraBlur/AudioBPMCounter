# 2023-11-20 V 1.0
# 最后修改： 高胡宇晨
# 判断阈值需要修改，现在版本心率检测频率偏高
# OSC 输出部分尚未完成

import pyaudio
import time
import numpy as np
import scipy.signal as signal
from scipy.io import wavfile
from scipy.fft import fft, fftfreq
import os
import threading

def detectHeartBPM(audio_data, sample_rate):
    
    normalized_audio = audio_data / np.max(np.abs(audio_data))
    # Design a bandpass filter
    filter_order = 3
    low_freq = 20
    high_freq = 600
    b, a = signal.butter(filter_order, [low_freq, high_freq], btype='band', fs=sample_rate)

    filtered_audio = signal.filtfilt(b, a, audio_data)
    
    output_path = r'D:\Projects\Coding\OSC_BPMDetector\Audio\test'
    
    #output_wave_file(filtered_audio, output_path, count)
    #count += 1
    
    # Use peak detection to find peaks of the heartbeat signal
    peaks, _ = signal.find_peaks(filtered_audio, distance=sample_rate * 0.4)
    print("Peaks:", peaks)
    print("PeaksCount:", len(peaks))

    if len(peaks) > 1:
        # Calculate BPM
        bpm = 60 / np.mean(np.diff(peaks)) * sample_rate

        # Print the BPM result
        print("Heartbeat BPM:", bpm)
    else:
        print("No peaks found in the audio.")

def detectHeartBPM_periodogram(audio_data, sample_rate):
    # 执行频谱分析
    n = len(audio_data)  # 音频数据的长度
    timestep = 1 / sample_rate  # 采样间隔
    frequencies = fftfreq(n, timestep)  # 计算频率轴
    spectrum = fft(audio_data)  # 计算频谱

    # 提取心跳声频谱的特征频率范围
    heart_rate_range = (20, 400)  # 心跳声的频率范围
    heart_rate_mask = np.logical_and(frequencies >= heart_rate_range[0], frequencies <= heart_rate_range[1])

    # 在心跳声频谱范围内计算能量
    heart_rate_spectrum = np.abs(spectrum[heart_rate_mask])
    
    # 计算能量平均值作为阈值
    threshold = np.mean(heart_rate_spectrum)

    threshold = 800000
    
        # 设置最小心率间隔
    min_beat_interval = 0.5  # 0.5秒
    
    #检测心跳声
    if np.max(heart_rate_spectrum) > threshold:
            current_time = time.time()
            if current_time - detectHeartBPM_periodogram.last_beat_time >= min_beat_interval:
                print("Beat!")
                detectHeartBPM_periodogram.last_beat_time = current_time


    
def output_wave_file(audio_data, output_path, count):
    out_count = count
    output_path = os.path.join(output_path, f"{out_count:04d}.wav")
    export_dtype = np.int16  
    wavfile.write(output_path, 48000, audio_data.astype(export_dtype))

# 预处理音频，混合为MONO
def audio_pre(stream,  chunk_size):
    # Read audio data
    data1 = stream.read(chunk_size)
    audio_data = np.frombuffer(data1, dtype=np.int16)

        # Extract left and right channels
    left_channel = audio_data[::2]
    right_channel = audio_data[1::2]

    # Convert to mono by averaging the channels
    mono_audio = np.mean([left_channel, right_channel], axis=0)
        
    # Convert to signed 16-bit integer
    mono_audio = mono_audio.astype(np.int16)
    return mono_audio
        
        

def stream1process(chunk_size, interval, stream):
    while True:
        start_time = time.time()  # Record start time
        
        mono_audio1 = audio_pre(stream, chunk_size)
        
        detectHeartBPM(mono_audio1, sample_rate)
        # count += 1
        # print(f'\nCount:{count}')
        
        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time  # Calculate execution time
        if elapsed_time < interval:
            time.sleep(interval - elapsed_time)

def stream2process(chunk_size, interval, stream):
    while True:
        start_time = time.time()  # Record start time
        
        mono_audio1 = audio_pre(stream, chunk_size)

        detectHeartBPM_periodogram(mono_audio1, sample_rate)
        # count += 1
        # print(f'\nCount:{count}')

        end_time = time.time()  # Record end time
        elapsed_time = end_time - start_time  # Calculate execution time
        if elapsed_time < interval:
            time.sleep(interval - elapsed_time)
            
def main():
    interval1 = 10
    interval2 = 0.1
    count = 1
    global sample_rate
    sample_rate = 48000
    chunk_size1 = 48000 * interval1
    chunk_size2 = 48000 * interval2
    chunk_size1 = int(chunk_size1)
    chunk_size2 = int(chunk_size2)
    
    detectHeartBPM_periodogram.last_beat_time = 0.0  
    
    audio = pyaudio.PyAudio()

    stream1 = audio.open(format=pyaudio.paInt16,
                        channels=2,  # Stereo
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size1)
    
    stream2 = audio.open(format=pyaudio.paInt16,
                        channels=2,  # Stereo
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size2)

    # 创建并启动两个线程
    thread1 = threading.Thread(target=stream1process, args=(chunk_size1, interval1, stream1))
    thread2 = threading.Thread(target=stream2process, args=(chunk_size2, interval2, stream2)) 

    thread1.start()
    thread2.start()
    



if __name__ == '__main__':
    main()