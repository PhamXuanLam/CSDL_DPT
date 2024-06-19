import librosa
import numpy as np
import csv
import os

def calculate_average_amplitude(audio_segment):
    """
    Calculate average amplitude of an audio segment.
    
    Parameters:
        audio_segment (numpy.ndarray): Input audio segment.
    
    Returns:
        float: Average amplitude of the input audio segment.
    """
    # Calculate average amplitude
    average_amplitude = np.mean(np.abs(audio_segment))
    
    return average_amplitude

def calculate_pitch(audio_segment, sr):
    """
    Extract pitch (fundamental frequency) from an audio segment.
    
    Parameters:
        audio_segment (numpy.ndarray): Input audio segment.
        sr (int): Sampling rate of the audio segment.
    
    Returns:
        float: Pitch (fundamental frequency) of the input audio segment.
    """
    # Extract pitch using librosa
    pitches, magnitudes = librosa.core.piptrack(y=audio_segment, sr=sr)
    pitch_mean = np.mean(pitches)
    
    return pitch_mean

def calculate_loudness(audio_segment):
    """
    Calculate loudness of an audio segment.

    Parameters:
        audio_segment (numpy.ndarray): Input audio segment.

    Returns:
        float: Loudness of the input audio segment (in decibels).
    """
    # Calculate the root mean square (RMS) amplitude
    rms_amplitude = np.sqrt(np.mean(np.square(audio_segment)))

    # Convert RMS amplitude to decibels (assuming 0 dB reference level)
    loudness = 20 * np.log10(rms_amplitude)

    return loudness

def calculate_zero_crossing_rate(audio_segment):
    """
    Calculate zero crossing rate of an audio segment.
    
    Parameters:
        audio_segment (numpy.ndarray): Input audio segment.
    
    Returns:
        float: Zero crossing rate of the input audio segment.
    """
    # Calculate zero crossing rate
    zero_crossings = librosa.zero_crossings(audio_segment, pad=False)
    zero_crossing_rate = sum(zero_crossings) / len(zero_crossings)
    
    return zero_crossing_rate

def calculate_silent_ratio(audio_segment, threshold=0.01):
    """
    Calculate silent ratio of an audio segment.

    Parameters:
        audio_segment (numpy.ndarray): Input audio segment.
        threshold (float): Threshold amplitude value to determine silence (default: 0.01).

    Returns:
        float: Silent ratio of the input audio segment.
    """
    # Count the number of samples below the threshold
    silent_samples = sum(1 for sample in audio_segment if abs(sample) < threshold)

    # Calculate the silent ratio
    total_samples = len(audio_segment)
    silent_ratio = silent_samples / total_samples

    return silent_ratio

def split_audio(audio_file, window_size_ms=1000, overlap_ratio=0.5):
    # Đọc tập tin âm thanh
    y, sr = librosa.load(audio_file)

    # Tính kích thước cửa sổ và bước nhảy
    window_size = int(window_size_ms / 1000 * sr)
    overlap_size = int(window_size * overlap_ratio)

    # Chia nhỏ âm thanh thành các đoạn chồng lấn nhau
    segments = []
    start = 0
    while start < len(y):
        end = min(start + window_size, len(y))
        segments.append(y[start:end])
        start += (window_size - overlap_size)

    return segments, sr

def save_features_to_csv(features_list, audio_file_path, csv_file):
    # Kiểm tra xem file CSV đã tồn tại chưa
    file_exists = os.path.isfile(csv_file)

    # Mở file CSV để ghi (mode='a' để ghi tiếp vào file nếu file đã tồn tại)
    with open(csv_file, mode='a', newline='') as file:
        # Xác định các trường (columns) trong file CSV
        fieldnames = ['Audio_File', 'Amplitudes', 'Pitches', 'Loudness', 'Zero_Crossing_Rate', 'Silent_Ratios']
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        # Nếu file CSV mới được tạo ra, viết tiêu đề
        if not file_exists:
            writer.writeheader()

        # Chuyển mảng thành chuỗi trước khi ghi vào file CSV
        writer.writerow({
            'Audio_File': audio_file_path,
            'Amplitudes': ','.join(map(str, features_list['Amplitudes'])),
            'Pitches': ','.join(map(str, features_list['Pitches'])),
            'Loudness': ','.join(map(str, features_list['Loudness'])),
            'Zero_Crossing_Rate': ','.join(map(str, features_list['Zero_Crossing_Rate'])),
            'Silent_Ratios': ','.join(map(str, features_list['Silent_Ratios']))
        })

def read_features_from_csv(csv_file):
    features_list = []
    
    with open(csv_file, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            features = {
                'Audio_File': row['Audio_File'],
                'Amplitudes': [float(x) for x in row['Amplitudes'].split(',')],
                'Pitches': [float(x) for x in row['Pitches'].split(',')],
                'Loudness': [float(x) for x in row['Loudness'].split(',')],
                'Zero_Crossing_Rate': [float(x) for x in row['Zero_Crossing_Rate'].split(',')],
                'Silent_Ratios': [float(x) for x in row['Silent_Ratios'].split(',')]
            }
            features_list.append(features)
    
    return features_list

def save_audio_features_to_csv(folder_path):
    """
    Read audio files in a folder and return segments along with their sampling rates.

    Parameters:
        folder_path (str): Path to the folder containing audio files.
    """

    # Iterate over audio files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            audio_file = os.path.join(folder_path, filename)
            
            segments, sr = split_audio(audio_file)
            
            amplitude_list = []
            pitch_list = []
            loudness_list = []
            silent_ratio_list = []
            zero_crossing_rate_list = []

            features_list = {}

            for segment in segments:
                amplitude = calculate_average_amplitude(segment)
                pitch = calculate_pitch(segment, sr)
                loudness = calculate_loudness(segment)
                silent_ratio = calculate_silent_ratio(segment)
                zero_crossing_rate = calculate_zero_crossing_rate(segment)
                amplitude_list.append(amplitude)
                pitch_list.append(pitch)
                loudness_list.append(loudness)
                silent_ratio_list.append(silent_ratio)
                zero_crossing_rate_list.append(zero_crossing_rate)

            # Gán các danh sách đặc trưng vào features_list
            features_list["Amplitudes"] = amplitude_list
            features_list["Pitches"] = pitch_list
            features_list["Loudness"] = loudness_list
            features_list["Silent_Ratios"] = silent_ratio_list
            features_list["Zero_Crossing_Rate"] = zero_crossing_rate_list
            
            save_features_to_csv(features_list, audio_file, "audio_features.csv")

def save_audio_features_to_csv(folder_path):
    """
    Read audio files in a folder and return segments along with their sampling rates.

    Parameters:
        folder_path (str): Path to the folder containing audio files.
    """

    # Iterate over audio files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            audio_file = os.path.join(folder_path, filename)
            
            segments, sr = split_audio(audio_file)
            
            amplitude_list = []
            pitch_list = []
            loudness_list = []
            silent_ratio_list = []
            zero_crossing_rate_list = []

            features_list = {}

            for segment in segments:
                amplitude = calculate_average_amplitude(segment)
                pitch = calculate_pitch(segment, sr)
                loudness = calculate_loudness(segment)
                silent_ratio = calculate_silent_ratio(segment)
                zero_crossing_rate = calculate_zero_crossing_rate(segment)
                amplitude_list.append(amplitude)
                pitch_list.append(pitch)
                loudness_list.append(loudness)
                silent_ratio_list.append(silent_ratio)
                zero_crossing_rate_list.append(zero_crossing_rate)

            # Gán các danh sách đặc trưng vào features_list
            features_list["Amplitudes"] = amplitude_list
            features_list["Pitches"] = pitch_list
            features_list["Loudness"] = loudness_list
            features_list["Silent_Ratios"] = silent_ratio_list
            features_list["Zero_Crossing_Rate"] = zero_crossing_rate_list
            
            save_features_to_csv(features_list, audio_file, "audio_features.csv")

# Example usage:
# segments_with_sr = read_audio_files("folder_path")
