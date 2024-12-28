# Copyright (c) 2024 Umang Bansal
# 
# This software is the property of Umang Bansal. All rights reserved.
# Unauthorized copying of this file, via any medium, is strictly prohibited.
# Licensed under the MIT License. See the LICENSE file for details.

import serial
import numpy as np
from scipy import signal
import pandas as pd
import time
import pickle
import pyautogui
import keyboard

from collections import deque 

# Suppress sklearn warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



def setup_filters(sampling_rate):
    b_notch, a_notch = signal.iirnotch(50.0 / (0.5 * sampling_rate), 30.0)
    b_bandpass, a_bandpass = signal.butter(4, [0.5 / (0.5 * sampling_rate), 30.0 / (0.5 * sampling_rate)], 'band')
    return b_notch, a_notch, b_bandpass, a_bandpass

def process_eeg_data(data, b_notch, a_notch, b_bandpass, a_bandpass):
    data = signal.filtfilt(b_notch, a_notch, data)
    data = signal.filtfilt(b_bandpass, a_bandpass, data)
    return data

def calculate_psd_features(segment, sampling_rate):
    f, psd_values = signal.welch(segment, fs=sampling_rate, nperseg=len(segment))
    bands = {'alpha': (8, 13), 'beta': (14, 30), 'theta': (4, 7), 'delta': (0.5, 3)}
    features = {}
    for band, (low, high) in bands.items():
        idx = np.where((f >= low) & (f <= high))
        features[f'E_{band}'] = np.sum(psd_values[idx])
    features['alpha_beta_ratio'] = features['E_alpha'] / features['E_beta'] if features['E_beta'] > 0 else 0
    return features

def calculate_additional_features(segment, sampling_rate):
    f, psd = signal.welch(segment, fs=sampling_rate, nperseg=len(segment))
    peak_frequency = f[np.argmax(psd)]
    spectral_centroid = np.sum(f * psd) / np.sum(psd)
    log_f = np.log(f[1:])
    log_psd = np.log(psd[1:])
    spectral_slope = np.polyfit(log_f, log_psd, 1)[0]
    return {'peak_frequency': peak_frequency, 'spectral_centroid': spectral_centroid, 'spectral_slope': spectral_slope}

def load_model_and_scaler():
    with open(r'C:\Users\yadny\OneDrive\Desktop\Integrated\modelfinal.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open(r'C:\Users\yadny\OneDrive\Desktop\Integrated\scalerfinal.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return clf, scaler

def EEG():
    count_zero=0
    count_one=0
    ser = serial.Serial('COM5', 115200, timeout=1)
    clf, scaler = load_model_and_scaler()
    b_notch, a_notch, b_bandpass, a_bandpass = setup_filters(512)
    buffer = deque(maxlen=512)  

    while True:
        # try:
            if keyboard.is_pressed('q'):
                print("Stopping recording...")
                break
            raw_data = ser.readline().decode('latin-1').strip()
            if raw_data:
                eeg_value = float(raw_data)
                buffer.append(eeg_value)
               

                if len(buffer) == 512:
                    buffer_array = np.array(buffer)
                    processed_data = process_eeg_data(buffer_array, b_notch, a_notch, b_bandpass, a_bandpass)
                    psd_features = calculate_psd_features(processed_data, 512)
                    additional_features = calculate_additional_features(processed_data, 512)
                    features = {**psd_features, **additional_features}
                    
                    df = pd.DataFrame([features])
                    X_scaled = scaler.transform(df)
                    prediction = clf.predict(X_scaled)
                    print(type(prediction[0]))
                    print(f"Predicted Class: {prediction}")
                    if prediction == 1.:
                        count_one+=1
                    elif prediction==0.:
                        count_zero+=1    

                    # print(count_zero,count_one)
                    buffer.clear()
                    # if prediction == 0:
                    #     pyautogui.keyDown('space')
                    #     time.sleep(1)  # You can adjust the duration the key is pressed
                    #     pyautogui.keyUp('space')

                    # elif prediction == 1:
                    #     pyautogui.keyDown('w')
                    #     time.sleep(1)  # You can adjust the duration the key is pressed
                    #     pyautogui.keyUp('w') 
                    

        # except Exception as e:
        #     print(f'Error: {e}')
        #     continue
    # if int(count_zero)>int(count_one):
    #     print("Concentration")
    #     emot="Concentration"
 
    # else:
    #     print("Relaxation") 
    #     emot="Relaxation" 
    return count_one,count_zero 

                
                
        


    