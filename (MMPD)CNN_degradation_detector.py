#!/usr/bin/env python
# coding: utf-8

import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sg
import os
import time
import glob
import gc
import h5py
import math
import random
from tensorflow import config
from tensorflow.keras import backend
from tensorflow.keras import applications
from tensorflow.keras.models import Model, model_from_json
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from datetime import datetime
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score

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
def get_melspec(folder_path, binary_label, audiolen, frame_length, frame_shift, Mel_scale, Mel_cutf, Mel_channel, Mel_norm, VAD_drop):
    
    #Inicialize list
    x = []
    y = []
    
    #Get .wav files as an object
    files = glob.glob(folder_path + "/*.wav")
    print("Folder:" + folder_path)
    
    #For a progress bar
    nfiles = len(files)
    unit = math.floor(nfiles/20)
    bar = "#" + " " * math.floor(nfiles/unit)
    
    #Repeat every file-name
    for i, file in enumerate(files):
        
        #Display a progress bar
        print("\rProgress:[{0}] {1}/{2} Processing...".format(bar, i+1, nfiles), end="")
        if i % unit == 0:
            bar = "#" * math.ceil(i/unit) + " " * math.floor((nfiles-i)/unit)
            print("\rProgress:[{0}] {1}/{2} Processing...".format(bar, i+1, nfiles), end="")
        
        #Read the .wav file
        data, Fs = sf.read(file)
        #Transform stereo into monoral
        if(isinstance(data[0], list) == True):
            wavdata = 0.5*data[:, 0] + 0.5*data[:, 1]
        else:
            wavdata = data
        
        #Down sampling and normalization of the wave
        #wavdata = sg.resample_poly(wavdata, 8000, Fs)
        #Fs = 8000
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
            
            #Add results to list sequentially
            x.append(melspec)
            if binary_label == 0 or binary_label ==1 :
                y.append(binary_label)
        
        #In case of audio is shorter than audiolen
        else:
            print("\rAudio file:" + file + " has been skipped.\nBecause the audio is shorter than audiolen.\n")
    
    #Finish the progress bar
    bar = "#" * math.ceil(nfiles/unit)
    print("\rProgress:[{0}] {1}/{2} Completed!   ".format(bar, i+1, nfiles), end="")
    print()
    
    #Convert into numpy array
    x = np.array(x)
    y = np.array(y)
    
    #Return the result
    return x, y

### Function to change the learning rate for each epoch ###
def step_decay(x):
    y = learn_rate * 10**(-lr_decay*x)
    return y

### Function for executing CNN learning ###
def CNN_learning(train_x, train_y, test_x, test_y, detect_label, LR, BS, EP, log_path, fold, mode):
    
    #Memory saving
    devices = config.experimental.list_physical_devices('GPU')
    if len(devices) > 0:
        for k in range(len(devices)):
            config.experimental.set_memory_growth(devices[k], True)
            print('memory growth:', config.experimental.get_memory_growth(devices[k]))
    else:
        print("Not enough GPU hardware devices available")
    
    #Calculate the 1st and 2nd derivative (file, time, Mel-frequency, derivative)
    diff1 = np.diff(test_x, n=1, axis=1)[:, 1:, :, np.newaxis] #Trim the data for corresponding to 2nd derivative
    diff2 = np.diff(test_x, n=2, axis=1)[:, :, :, np.newaxis]
    test_X = test_x[:, 2:, :, np.newaxis] #Trim the data for corresponding to 2nd derivative
    test_X = np.concatenate([test_X, diff1], 3)
    test_X = np.concatenate([test_X, diff2], 3)
    
    #Delete the valuables to save memory
    del diff1
    del diff2
    del test_x
    gc.collect()
    
    #Path for saving CNN model
    p1 = "./models/MMPD_pathological_voice/" + detect_label + "_" + str(fold+1) + "model.json"
    p2 = "./models/MMPD_pathological_voice/" + detect_label + "_" + str(fold+1) + "weights.h5"
    
    #In case of existing pre-learned model
    if os.path.isfile(p1) and os.path.isfile(p2) and mode == 1:
        #Read the pre-learned model
        with open(p1, "r") as f:
            cnn_model = model_from_json(f.read())
            cnn_model.load_weights(p2)
    
    #In case of learning from the beginning
    else:
        #Calculate the 1st and 2nd derivative (file, time, Mel-frequency, derivative)
        diff1 = np.diff(train_x, n=1, axis=1)[:, 1:, :, np.newaxis] #Trim the data for corresponding to 2nd derivative
        diff2 = np.diff(train_x, n=2, axis=1)[:, :, :, np.newaxis]
        train_X = train_x[:, 2:, :, np.newaxis] #Trim the data for corresponding to 2nd derivative
        train_X = np.concatenate([train_X, diff1], 3)
        train_X = np.concatenate([train_X, diff2], 3)
        
        #Delete the valuables to save memory
        del diff1
        del diff2
        del train_x
        gc.collect()
        
        #Get the number of row and column in Mel-spectrogram
        row = train_X.shape[1]
        column = train_X.shape[2]
        #print("input_data_shape: " + str(train_X.shape) )
        
        #Define the input size(row, column, color)
        image_size = Input(shape=(row, column, 3))
        
        #Construct the CNN model with Functional API by Keras
        x = BatchNormalization()(image_size)
        x = Conv2D(32, (3, 3), padding='same', activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(32, (3, 3), padding='same', activation="relu")(x)
        x = BatchNormalization()(x)
        #x = Conv2D(32, (3, 3), padding='same', activation="relu")(x)
        #x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same', activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(64, (3, 3), padding='same', activation="relu")(x)
        x = BatchNormalization()(x)
        #x = Conv2D(64, (3, 3), padding='same', activation="relu")(x)
        #x = BatchNormalization()(x)
        x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        x = Conv2D(128, (3, 3), padding='same', activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), padding='same', activation="relu")(x)
        x = BatchNormalization()(x)
        x = Conv2D(128, (3, 3), padding='same', activation="relu")(x)
        x = BatchNormalization()(x)
        #x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        #x = Conv2D(256, (3, 3), padding='same', activation="relu")(x)
        #x = BatchNormalization()(x)
        #x = Conv2D(256, (3, 3), padding='same', activation="relu")(x)
        #x = BatchNormalization()(x)
        #x = Conv2D(256, (3, 3), padding='same', activation="relu")(x)
        #x = BatchNormalization()(x)
        #x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        #x = Conv2D(512, (3, 3), padding='same', activation="relu")(x)
        #x = BatchNormalization()(x)
        #x = Conv2D(512, (3, 3), padding='same', activation="relu")(x)
        #x = BatchNormalization()(x)
        #x = Conv2D(512, (3, 3), padding='same', activation="relu")(x)
        #x = BatchNormalization()(x)
        x = GlobalAveragePooling2D()(x)
        #x = Flatten()(x)
        #x = Dense(256, activation='relu')(x)
        x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
        x = BatchNormalization()(x)
        x = Dense(1, activation='sigmoid')(x)
        
        #Construct the model and display summary
        cnn_model = Model(image_size, x)
        #print(cnn_model.summary())
        
        #Define the optimizer (SGD with momentum or Adam)
        opt = SGD(lr=LR, momentum=0.9, decay=0.0)
        #opt = Adam(lr=LR, beta_1=0.9, beta_2=0.999)
        
        #Compile the model
        cnn_model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
        
        #Start learning
        lr_decay = LearningRateScheduler(step_decay)
        hist = cnn_model.fit(train_X, train_y, batch_size=BS, epochs=EP, validation_data=(test_X, test_y), callbacks=[lr_decay], verbose=1)
        
        #Save the learned model
        model_json = cnn_model.to_json()
        with open(p1, 'w') as f:
            f.write(model_json)
        cnn_model.save_weights(p2)
        
        #Save the learning history as text file
        loss = hist.history['loss']
        acc = hist.history['acc']
        val_loss = hist.history['val_loss']
        val_acc = hist.history['val_acc']
        with open(log_path, "a") as fp:
            fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
            for i in range(len(acc)):
                fp.write("%d\t%f\t%f\t%f\t%f" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))
                fp.write("\n")
        
        #Display the learning history
        plt.rcParams.update({'font.size': 14})
        fig, (axL, axA) = plt.subplots(ncols=2, figsize=(18, 5))
        #Loss function
        axL.plot(hist.history['loss'], label="loss for training")
        axL.plot(hist.history['val_loss'], label="loss for validation")
        axL.set_title('model loss')
        axL.set_xlabel('epoch')
        axL.set_ylabel('loss')
        axL.legend(loc='upper right')
        #Score
        axA.plot(hist.history['acc'], label="accuracy for training")
        axA.plot(hist.history['val_acc'], label="accuracy for validation")
        axA.set_title('model accuracy')
        axA.set_xlabel('epoch')
        axA.set_ylabel('accuracy')
        axA.legend(loc='lower right')
        #plt.show()
        #Save the graph
        fig.savefig("./models/MMPD_pathological_voice/" + detect_label + "_" + str(fold+1) + "loss_accuracy.png")
    
    #Get the score for evaluation data
    proba_y = cnn_model.predict(test_X)
    
    #Restart the session to relieve the GPU memory (to prevent Resource Exhausted Error)
    backend.clear_session()
    #backend.get_session() #Less than tensorflow ver.1.14
    del cnn_model
    gc.collect()

    #Sleep 1 minute for cooling down the GPU
    #time.sleep(60)
    
    #Return the learning history and binary score
    return proba_y

### Function for calculating AUC(Area Under ROC Curve) and its standard error ###
def get_AUC(test_y, proba_y, detect_label, fold):
    
    #Compute the AUC
    AUC = roc_auc_score(test_y, proba_y)
    
    #Plot the ROC curve
    #plt.rcParams["font.size"] = 16
    #plt.figure(figsize=(10, 6), dpi=100)
    #fpr, tpr, thresholds = roc_curve(test_y, proba_y)
    #plt.plot([0, 1], [0, 1], linestyle='--')
    #plt.plot(fpr, tpr, marker='.')
    #plt.title('ROC curve')
    #plt.xlabel('False positive rate')
    #plt.ylabel('True positive rate')
    #plt.savefig("./log/" + detect_label + "_" + str(fold+1) + "ROCcurve.png")
    #plt.show()
    
    #Return AUC
    return AUC

### Main ###
if __name__ == "__main__":
    
    #Set up
    audiolen = 6           #Cropping time for audio (second) [Default]6
    frame_length = 0.03    #STFT window width (second) [Default]0.03
    frame_shift = 0.02     #STFT window shift (second) [Default]0.02
    Mel_scale = 700        #Mel-frequency is proportional to "log(f/Mel_scale + 1)" [Default]700
    Mel_cutf = [0, None]   #The cutoff frequency (Hz, Hz) of Mel-filter [Default] [0, None(=Nyquist)]
    Mel_channel = 40       #The number of frequency channel for Mel-spectrogram [Default]40
    Mel_norm = False       #Normalize the area underneath each Mel-filter into 1 [Default]False
    VAD_drop=False         #Drop non-speech frames by voice activity detection
    detect_label = "NZ"    #Degradation what to detect (DT, NZ, RV)
    learn_rate = 1e-2      #Lerning rate for CNN training [Default]1e-2
    lr_decay = 0.1         #Lerning rate is according to "learn_rate*10**(-lr_decay*n_epoch)" [Default]0.1
    batch_size = 64        #Size of each batch for CNN training [Default]64
    epoch = 20             #The number of repeat for CNN training [Default]20
    cv = 10                #The number of folds for cross varidation [Default]10
    Melmode = 0            #0: calculate mel from the beginning, 1: read local files [Default]0
    CNNmode = 0            #0: train from the beginning, 1: read pre-learned model [Default]0 
    
    #In case of calculating the Mel-Spectrogram from audio
    if Melmode == 0:
        #Define the class names for ***training data***
        classes = ['CLEAN', 'DT', 'NZ', 'RV', 'NR']
        if detect_label == "CLEAN":
            binary_labels = [1, 0, 0, 0, 0]
        elif detect_label == "DT":
            binary_labels = [0, 1, 0, 0, 0]
        elif detect_label == "NZ":
            binary_labels = [0, 0, 1, 0, 1]
        elif detect_label == "RV":
            binary_labels = [0, 0, 0, 1, 1]
        
        #Call my function for calculating Mel-spectrogram
        for i, cl in enumerate(classes):
            fpath = "./audio_data/MMPD_pathological_voice/training/" + cl
            x, y = get_melspec(fpath, binary_labels[i], audiolen, frame_length, frame_shift, Mel_scale, Mel_cutf, Mel_channel, Mel_norm, VAD_drop)
            if i == 0:
                train_x, train_y = x, y
            else:
                train_x = np.concatenate((train_x, x), axis=0)
                train_y = np.concatenate((train_y, y), axis=0)
        
        #Save the training data
        fpath = "./numpy_files/MMPD_pathological_voice/training"
        np.save(fpath + '/X_' + detect_label + 'train', train_x)
        np.save(fpath + '/Y_' + detect_label + 'train', train_y)
        
        #Define the class names for ***evaluation data***
        classes = ['CLEAN', 'DT', 'NZ', 'RV', 'OTHERS', 'NR']
        if detect_label == "CLEAN":
            binary_labels = [1, 0, 0, 0, 0, 0]
        elif detect_label == "DT":
            binary_labels = [0, 1, 0, 0, 0, 0]
        elif detect_label == "NZ":
            binary_labels = [0, 0, 1, 0, 0, 1]
        elif detect_label == "RV":
            binary_labels = [0, 0, 0, 1, 0, 1]
        
        #Repeat every classes
        test_xs = []
        for i, cl in enumerate(classes):
            #Calculating Mel-spectrogram of true-class data
            fpath = "./audio_data/MMPD_pathological_voice/evaluation/" + cl
            x, y = get_melspec(fpath, binary_labels[i], audiolen, frame_length, frame_shift, Mel_scale, Mel_cutf, Mel_channel, Mel_norm, VAD_drop)
            
            #Save the test data
            fpath = "./numpy_files/MMPD_pathological_voice/evaluation"
            np.save(fpath + '/X_' + cl + 'test', x)
            test_xs.append(x) #test data is used for cross validation
    
    #In case of reading the Mel-spectrogram from local file
    else:
        #Read the training data and evaluation data
        fpath = "./numpy_files/MMPD_pathological_voice/training"
        train_x = np.load(fpath + '/X_' + detect_label + 'train' + '.npy')
        train_y = np.load(fpath + '/Y_' + detect_label + 'train' + '.npy')
        
        #Define the class names for ***valuation data***
        classes = ['CLEAN', 'DT', 'NZ', 'RV', 'OTHERS', 'NR']
        
        #Repeat every classes
        test_xs = []
        for i, cl in enumerate(classes):
            #Read the data from local file
            fpath = "./numpy_files/MMPD_pathological_voice/evaluation"
            x = np.load(fpath + '/X_' + cl + 'test.npy')
            test_xs.append(x) #test data is used for cross validation
    
    #Standardize the input data
    ave = np.average(train_x, axis=None)
    std = np.std(train_x, axis=None)
    train_x = (train_x - ave)/std
    for i in range(len(test_xs)):
        test_xs[i] = (test_xs[i] - ave)/std
    
    #Prepare for process-log
    message = "Training for " + detect_label + "-detector\n\n"
    log_path = "./log/" + detect_label + "_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
    with open(log_path, "w") as fp:
        fp.write(message)
    
    #Define parameters for cross validation
    true_cl = classes.index(detect_label)
    total_files = test_xs[0].shape[0]
    test_files = math.floor(total_files / cv)
    enu = list(range(total_files))
    
    #Initialize the vector for AUC score and Detection-Rate
    AUC_vector = np.zeros(cv)
    DR_vector = np.zeros(cv)
    
    #Repeat every fold
    for fold in range(cv):
        
        #Get randomly test sampling without replacement
        test_i = random.sample(enu, k=test_files)
        train_i = list(set(enu) - set(test_i)) #The remain is for training
        
        #Get the test data and class label for true-class
        true_x = test_xs[true_cl]
        test_x = true_x[test_i]
        test_y = np.ones(test_files, dtype=np.int) #True-class=1
        
        #Remain test data is used as training data
        plus_x = true_x[train_i]
        plus_y = np.ones(plus_x.shape[0], dtype=np.int)
        train_X = np.concatenate((train_x, plus_x), axis=0)
        train_Y = np.concatenate((train_y, plus_y), axis=0)
        
        #Construct test data (True-Class + Others)
        for cl in range(6):
            
            #Extend the test data except for true-class
            if cl != true_cl:
                #For outlier
                if cl == 4:
                    x = test_xs[cl]
                    outlier_i = list(range(50, 50+test_files)) #Indices are fixed
                    x = x[outlier_i]
                #For other than outlier
                else:
                    x = test_xs[cl]
                    x = x[test_i]
                test_x = np.concatenate((test_x, x), axis=0)
        
        #Construct class label (True-Class=1, NR=0 or 1, Others=0)
        for cl in range(6):
            
            #Extend the class label except for true-class
            if cl != true_cl:
                #For NR
                if cl == 5:
                    if true_cl == 2 or true_cl == 3: #In the case of NZ or RV
                        y = np.ones(test_files, dtype=np.int)
                    else: #In the case of CLEAR or DT
                        y = np.zeros(test_files, dtype=np.int)
                #For other than NR
                else:
                    y = np.zeros(test_files, dtype=np.int)
                test_y = np.concatenate([test_y, y], axis=0)
        
        #Get the start time
        start = time.time()
        
        #Call my function for executing CNN learning (train_X = train_x + plus_x dataset)
        proba_y = CNN_learning(train_X, train_Y, test_x, test_y, detect_label, learn_rate, batch_size, epoch, log_path, fold, CNNmode)
        
        #Call my function for calculating the AUC
        A = get_AUC(test_y, proba_y, detect_label, fold)
        AUC_vector[fold] = A
        
        #Output the binary accuracy (Detection Rate)
        pred_y = np.where(proba_y < 0.5, 0, 1) #Binary threshold = 0.5
        B = accuracy_score(test_y, pred_y)
        DR_vector[fold] = B
        print(classification_report(test_y, pred_y))
        
        #Construct the process log
        finish = time.time() - start
        report = "Fold{}: AUC_{}={:.5f}, Detection_rate={:.5f}, Process_time={:.1f}sec\n".format(fold+1, classes[true_cl], A, B, finish)
        message = message + report
        print(report)
    
    #Average the result of cv-folds
    AUC = np.average(AUC_vector)
    DR = np.average(DR_vector)
    SE_AUC = np.std(AUC_vector) / np.sqrt(cv-1) #Population variance in cross varidation
    SE_DR = np.std(DR_vector) / np.sqrt(cv-1)
    
    #Output the result
    report = "AUC_{}={:.5f}, CI(95%)=±{:.5f}, Detection_rate={:.5f}, CI(95%)=±{:.5f}".format(detect_label, AUC, 1.96*SE_AUC, DR, 1.96*SE_DR)
    message = message + report
    print(report)
    with open(log_path, "a") as fp:
        fp.write(message)