#!/usr/bin/env python
# coding: utf-8

import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sg
import os
import sys
import time
import glob
import gc
import h5py
import math
import random
from PIL import Image
import cv2
from tensorflow.keras.preprocessing import image as images
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras import backend
from tensorflow.keras.models import Model, model_from_json

### Function for generating Mel-scale filters ###
def melFilterBank(Fs, fftsize, Mel_scale, Mel_cutf, Mel_channel, Mel_norm):
    
    #Define Mel-scale parameter m0 based on "1000Mel = 1000Hz"
    m0 = 1000.0 / np.log(1000.0 / Mel_scale + 1.0)
    
    #Resolution of frequency
    df = Fs / fftsize
    
    #Mel-scale filters are periodic triangle-shaped structures
    #Define the lower and higher frequency limit of Mel-scale filers
    Nyq = Fs / 2
    f_low, f_high = Mel_cutf
    if f_low is None:
        f_low = 0
    elif f_low < 0:
        f_low = 0
    if f_high is None:
        f_high = Nyq
    elif f_high > Nyq or f_high <= f_low: 
        f_high = Nyq
    #Convert into Mel-scale
    mel_Nyq = m0 * np.log(Nyq / Mel_scale + 1.0)
    mel_low = m0 * np.log(f_low / Mel_scale + 1.0)
    mel_high = m0 * np.log(f_high / Mel_scale + 1.0)
    #Convert into index-scale
    n_Nyq = round(fftsize / 2)
    n_low = round(f_low / df)
    n_high = round(f_high / df)
    
    #Calculate the Mel-scale interval between triangle-shaped structures
    #Divided by channel+1 because the termination is not the center of triangle but its right edge
    dmel = (mel_high - mel_low) / (Mel_channel + 1)
    
    #List up the center position of each triangle
    mel_center = mel_low + np.arange(1, Mel_channel + 1) * dmel
    
    #Convert the center position into Hz-scale
    f_center = Mel_scale * (np.exp(mel_center / m0) - 1.0)
    
    #Define the center, start, and end position of triangle as index-scale
    n_center = np.round(f_center / df)
    n_start = np.hstack(([n_low], n_center[0 : Mel_channel - 1]))
    n_stop = np.hstack((n_center[1 : Mel_channel], [n_high]))
    
    #Initial condition is defined as 0 padding matrix
    output = np.zeros((n_Nyq, Mel_channel))
    
    #Repeat every channel
    for c in np.arange(0, Mel_channel):
        
        #Slope of a triangle(growing slope)
        upslope = 1.0 / (n_center[c] - n_start[c])
        
        #Add a linear function passing through (nstart, 0) to output matrix 
        for x in np.arange(n_start[c], n_center[c]):
            #Add to output matrix
            x = int(x)
            output[x, c] = (x - n_start[c]) * upslope
        
        #Slope of a triangle(declining slope)
        dwslope = 1.0 / (n_stop[c] - n_center[c])
        
        #Add a linear function passing through (ncenter, 1) to output matrix 
        for x in np.arange(n_center[c], n_stop[c]):
            #Add to output matrix
            x = int(x)
            output[x, c] = 1.0 - ((x - n_center[c]) * dwslope)
        
        #Normalize area underneath each Mel-filter into 1
        #[ref] T.Ganchev, N.Fakotakis, and G.Kokkinakis, Proc. of SPECOM 1, 191-194 (2005)
        #https://pdfs.semanticscholar.org/f4b9/8dbd75c87a86a8bf0d7e09e3ebbb63d14954.pdf
        if Mel_norm == True:
            output[:, c] = output[:, c] * 2 / (n_stop[c] - n_start[c])
    
    #Return Mel-scale filters as list (row=frequency, column=Mel channel)
    return output

### Function for getting speech frames ###
def energy_based_VAD(wavdata, FL, FS):
    
    #Construct the frames
    nframes = 1 + np.int(np.floor((len(wavdata) - FL) / FS))
    frames = np.zeros((nframes, FL))
    for i in range(nframes):
        frames[i] = wavdata[i*FS : i*FS + FL]
    
    #Multiply the Hamming window
    HMW = sg.hamming(FL)
    HMW = HMW[np.newaxis, :]
    HMW = np.tile(HMW, (nframes, 1))
    frames = frames * HMW
    
    #Calculate the wave energy std
    S = 20 * np.log10(np.std(frames, axis=1) + 1e-9)
    maxS = np.amax(S)
    
    #Estimate the indices of speech frames
    VAD_index = np.where((S > maxS-30) & (S > -55))
    VAD_index = np.squeeze(np.array(VAD_index))
    
    return VAD_index

### Function for calculating Mel-Spectrogram ###
def get_melspec(file_path, audiolen, frame_length, frame_shift, Mel_scale, Mel_cutf, Mel_channel, Mel_norm, VAD_drop):
    
    #Read the audio file
    data, Fs = sf.read(file_path)
    print("Audio file:'" + file_path + "'.\nFs: " + str(Fs))
    if data.shape[0] >= audiolen*Fs:
        c = round(data.shape[0] / 2)
    
    #Transform stereo into monoral
    if(isinstance(data[0], list) == True):
        wavdata = 0.5*data[:, 0] + 0.5*data[:, 1]
    else:
        wavdata = data
    wavdata = (wavdata - np.mean(wavdata))
    wavdata = wavdata / np.amax(np.abs(wavdata))
    
    #Calculate the index of window size and overlap
    FL = round(frame_length * Fs)
    FS = round(frame_shift * Fs)
    OL = FL - FS
    
    #Call my function for getting speech frames
    VAD_index = energy_based_VAD(wavdata, FL, FS)
    
    #Pass through a pre-emphasis fileter to emphasize the high frequency
    wavdata = sg.lfilter([1.0, -0.97], 1, wavdata)
    
    #Execute STFT
    F, T, dft = sg.stft(wavdata, fs=Fs, window='hamm', nperseg=FL, noverlap=OL)
    Adft = np.abs(dft)[0 : round(FL/2)]**2
    
    #Call my function for generating Mel-scale filters(row: fftsize/2, column: Channel)
    filterbank = melFilterBank(Fs, FL, Mel_scale, Mel_cutf, Mel_channel, Mel_norm)
    
    #Multiply the filters into the STFT amplitude, and get logarithm of it
    melspec = Adft.T @ filterbank
    if np.any(melspec == 0):
        melspec = np.where(melspec == 0, 1e-9, melspec)
    melspec = np.log10(melspec)
    
    #Drop the non-speech frames
    if VAD_drop == True:
        melspec = melspec[VAD_index, :]
    
    #Cropping the melspec with length of audiolen
    if melspec.shape[0] >= audiolen/frame_shift:
        center = round(melspec.shape[0] / 2)
        melspec = melspec[round(center - audiolen/frame_shift/2) : round(center + audiolen/frame_shift/2), :]
    
    #In case of audio is shorter than audiolen
    else:
        print("Audio file:" + file_path + " has been skipped.\nBecause the audio is shorter than audiolen.")
        melspec = np.zeros(1)
    
    #Return the result
    return melspec

### Function for calculating Grad-CAM ###
def Grad_CAM(model, x, layer_name):
    
    #Change the learning phase into test mode
    backend.set_learning_phase(0)
        
    #Print the binary classification score
    print("Prediction score: " + str(model.predict(x)[0, 0]))
    
    #Get the original image size
    row = model.get_layer[0].output_shape[0][1]
    column = model.get_layer[0].output_shape[0][2]
    print("Input_size: " + str((row, column)))
    
    #Function to get final conv_output and gradients
    true_output = model.layers[-1].output #Output of the truely final layer
    mid_output = model.get_layer(layer_name).output  #Output of the final convolutional layer
    grads = backend.gradients(true_output, mid_output)[0]  #Calculate the "gradients(loss, variables)"
    mean_grads = backend.mean(grads, axis=(1, 2))  #Average the gradients
    gradient_function = backend.function([model.input], [mid_output, mean_grads])
    
    #Get the output of final conv_layer and the weight for each kernel (mean gradients)
    conv_output, kernel_weights = gradient_function([x])
    conv_output, kernel_weights = conv_output[0], kernel_weights[0]
    
    #Get the Class Activation Mapping (CAM)
    cam = conv_output @ kernel_weights
    #Caution! cv2-resize-shape is reverse of numpy-shape
    cam = cv2.resize(cam, (column, row), cv2.INTER_LINEAR) #Scale up
    cam = np.maximum(cam, 0) #We have no interest in negative value (like ReLu)
    cam = 255 * cam / cam.max()  #Normalize
    
    #Get together with original image
    original = x[0, :, :, 0][:, :, np.newaxis]  #Cut out the derivatives
    original = np.uint8(255 * (original - original.min()) / (original.max() - original.min()))
    heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_OCEAN)  #Add color to heat map
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  #Convert it into color map
    plusCAM = (np.float32(heatmap)*0.4 + original*0.6)  #Mix original image with heatmap
    plusCAM = np.uint8(255 * plusCAM / plusCAM.max())
    
    #Return the CAM data
    return original, heatmap, plusCAM

### Function for calculating Score-CAM ###
def Score_CAM(model, x, layer_name):
    
    #Print the binary classification score
    print("Prediction score: " + str(model.predict(x)[0, 0]))
    
    #Get the activation map for each kernel (Amap: A^k in original paper)
    Amap_array = Model(inputs=model.input, outputs=model.get_layer(layer_name).output).predict(x)
    
    #Get the original image size
    row = model.layers[0].output_shape[0][1]
    column = model.layers[0].output_shape[0][2]
    print("Input_size: " + str((row, column)))
    
    #Scale up the Amap into the original image size
    Amap_upsample_list = []
    for k in range(Amap_array.shape[3]):
        #Caution! cv2-resize-shape is reverse of numpy-shape
        Amap_upsample_list.append(cv2.resize(Amap_array[0,:,:,k], (column, row), cv2.INTER_LINEAR))
    
    #Normalize Amap into the range between 0 and 1
    Amap_norm_list = []
    for Amap_upsample in Amap_upsample_list:
        Amap_norm = Amap_upsample / (np.max(Amap_upsample) - np.min(Amap_upsample) + 1e-5)
        Amap_norm_list.append(Amap_norm)
    
    #Project into original image by multiplying the normalized Amap
    mask_input_list = []
    for Amap_norm in Amap_norm_list:
        mask_input = np.zeros_like(x) #initialize
        for c in range(3):
            mask_input[0,:,:,c] = x[0,:,:,c] * Amap_norm
        mask_input_list.append(mask_input)
    mask_input_array = np.concatenate(mask_input_list, axis=0)
    
    #Get the CNN output by masked input as weight for each kernel (S^k in original paper)
    kernel_weights = model.predict(mask_input_array)[:, 0]
    
    #Get the Class Activation Mapping (CAM)
    cam = Amap_array[0,:,:,:] @ kernel_weights
    #Caution! cv2-resize-shape is reverse of numpy-shape
    cam = cv2.resize(cam, (column, row), cv2.INTER_LINEAR) #Scale up
    cam = np.maximum(cam, 0) #We have no interest in negative value (like ReLu)
    cam = 255 * cam / cam.max() #Normalization
    
    #Get together with original image
    original = x[0, :, :, 0][:, :, np.newaxis]  #Cut out the derivatives
    original = np.uint8(255 * (original - original.min()) / (original.max() - original.min()))
    heatmap = cv2.applyColorMap(np.uint8(cam), cv2.COLORMAP_OCEAN)  #Add color to heat map
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  #Convert it into color map
    plusCAM = (np.float32(heatmap)*0.4 + original*0.6)  #Mix original image with heatmap
    plusCAM = np.uint8(255 * plusCAM / plusCAM.max())
    
    #Return the CAM data
    return original, heatmap, plusCAM

### Main ###
if __name__ == "__main__":
    
    #Set up
    frame_length = 0.03    #STFT window width (second) [Default]0.03
    frame_shift = 0.02     #STFT window shift (second) [Default]0.02
    Mel_scale = 700        #Mel-frequency is proportional to "log(f/Mel_scale + 1)" [Default]700
    Mel_cutf = [0, None]   #The cutoff frequency (Hz, Hz) of Mel-filters [Default] [0, None]
    Mel_channel = 40       #The number of frequency channel for Mel-spectrogram [Default]40
    Mel_norm = False       #Normalize the area underneath each Mel-filter into 1 [Default]False
    VAD_drop=False         #Drop non-speech frames by voice activity detection
    detect_label = "RV"    #Degradation what to detect (CLEAN, DT, NZ, RV)
    audio_label = "NR"     #Audio degradation type (CLEAN, DT, NZ, RV, NR, OTHERS)
    num_sample = 20        #The number of CAM images (randomly) [Default]20
    
    #Load the pre-learned model and its weight
    audiolen = 3       #Cropping time for audio (second)
    mymodel = "./models/Normal_running_speech"
    model_path = mymodel + "/" + detect_label + "_detector/" + detect_label + "_1model.json"
    weight_path = mymodel + "/" + detect_label + "_detector/" + detect_label + "_1weights.h5"
    if os.path.isfile(model_path) and os.path.isfile(weight_path):
        #Read the pre-learned model
        with open(model_path, "r") as f:
            cnn_model = model_from_json(f.read())
        cnn_model.load_weights(weight_path)
        backend.set_learning_phase(0)
        #print(cnn_model.summary())
    else:
        print("You input wrong path into p1 and p2.")
        os.sys.exit()
    
    #Extract the audio files randomly
    mydata = "./audio_data/Normal_running_speech"
    fpath = mydata + "/evaluation/" + audio_label
    files = glob.glob(fpath + "/*.wav")
    list_i = random.sample(list(range(len(files))), k=num_sample) #Extract samples randomly
    
    #Repeat for each audio
    for i in list_i:
        #Call my function for calculating Mel-spectrogram
        melspec = get_melspec(files[i], audiolen, frame_length, frame_shift, Mel_scale, Mel_cutf, Mel_channel, Mel_norm, VAD_drop)
        
        if melspec.all() != 0:
            
            #Standardize the input data based on mean and variance of training samples
            ave = -6.708185218079733
            std = 1.6368339571812376
            melspec = (melspec - ave)/std
            
            #Calculate the 1st and 2nd derivative (file, time, Mel-frequency, derivative)
            melspec = melspec[:, :, np.newaxis] #Add color axis
            diff1 = np.diff(melspec, n=1, axis=0)[1:, :, :] #Trim the data for corresponding to 2nd derivative
            diff2 = np.diff(melspec, n=2, axis=0)[:, :, :]
            x = melspec[2:, :, :] #Trim the data for corresponding to 2nd derivative
            x = np.concatenate([x, diff1], axis=2)
            x = np.concatenate([x, diff2], axis=2)
            x = x[np.newaxis, :, :, :]
            
            #Delete the valuables to save memory
            del diff1
            del diff2
            gc.collect()
            
            #Call for my function to get CAM image
            #original, Grad_heatmap, plusGradCAM = Grad_CAM(cnn_model, x, "conv2d_6")
            original, Score_heatmap, plusScoreCAM = Score_CAM(cnn_model, x, "conv2d_6")
            
            #Draw the mel-scale spectrogram of audio
            plt.rcParams["font.size"] = 8
            plt.figure(figsize=(5.3, 5), dpi=200) #Non-pathological speech setup
            
            #img_melspec = array_to_img(original.transpose(1,0,2))
            #plt.title('Mel-scale spectrogram')
            #plt.xlabel('Time [frames]')
            #plt.ylabel('Mel-scale frequency')
            #plt.xticks([50, 100, 150, 200, 250])
            #plt.yticks([10, 20, 30])
            #plt.imshow(img_melspec, cmap='gray')
            #plt.gca().invert_yaxis()
            
            #Draw the Score-CAM heatmap
            #img_heatmap = array_to_img(Score_heatmap)
            img_heatmap = array_to_img(plusScoreCAM.transpose(1,0,2))
            plt.title('Mel-scale spectrogram + Score-weighted CAM')
            plt.xlabel('Time [frames]')
            plt.ylabel('Mel-scale frequency')
            plt.xticks([50, 100, 150, 200, 250])
            plt.yticks([10, 20, 30])
            plt.imshow(img_heatmap)
            plt.gca().invert_yaxis()
            save_path = "./log/" + audio_label
            save_path = save_path + "_p" + files[i].split("p")[-1].replace(".wav", ".png")
            plt.savefig(save_path, dpi=200)
            
            #Draw the Grad-CAM heatmap
            #img_heatmap = array_to_img(Grad_heatmap)
            #img_heatmap = array_to_img(plusGradCAM.transpose(1,0,2))
            #plt.title('Gradient-weighted CAM')
            #plt.xlabel('Time [frames]')
            #plt.ylabel('Mel-scale frequency')
            #plt.xticks([50, 100, 150, 200, 250])
            #plt.yticks([10, 20, 30])
            #plt.imshow(img_heatmap)
            #plt.gca().invert_yaxis()