import pandas as pd
import numpy as np
import os
import json
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, cross_validate
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, Input, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
import pickle
from tqdm.notebook import tqdm_notebook as tq
from warnings import filterwarnings
filterwarnings('ignore')
import importlib
from library import faps_color as fapsc


def pad_curves(arr, new_shape1):

    pad_arr = np.zeros(arr.shape[0]*new_shape1).reshape(-1, new_shape1)
    pad_arr[:arr.shape[0], :arr.shape[1]] = arr
    print(pad_arr.shape)

    return pad_arr


def plot_error_hist(train_error, test_error, error_type, bin_num1, bin_num2, val=True, figsize=(8,4), dpi=80):

    fig, axs = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    #if error_type=="mae":
    #    fig.suptitle(f"Rekonstruktionsfehler in MAE", fontsize=18)
    #elif error_type=="mse":
    #    fig.suptitle(f"Rekonstruktionsfehler in MSE", fontsize=18)

    axs[0].hist(train_error, color=fapsc.dark_green, bins=bin_num1)
    axs[0].set_title("Trainingsdaten", fontsize=16)
    axs[0].set_xlabel("MAE", fontsize=16)
    axs[0].set_ylabel("Anzahl der Fehler", fontsize=16)
    axs[0].tick_params(axis='both', labelsize=14)

    axs[1].hist(test_error, color=fapsc.green, bins=bin_num2)
    if val:
        axs[1].set_title("Validierungsdaten", fontsize=16)
    else:
        axs[1].set_title("Tesdaten", fontsize=16)
    axs[1].set_xlabel("MAE", fontsize=16)
    axs[1].tick_params(axis='both', labelsize=14)

    plt.tight_layout(pad=1)
    plt.show()


def plot_clean_fraud(y_test, test_error, bin_num0, bin_num1, threshold, error_type, figsize=(10,5), dpi=80):

    clean = test_error[y_test==0]
    fraud = test_error[y_test>0]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.hist(clean, bins=bin_num0, width=0.006, label="Kurven der Klasse i.O.", alpha=1, color=fapsc.green)
    ax.hist(fraud, bins=bin_num1, label="Kurven der Fehlerklassen", alpha=0.5, color=fapsc.red)
    ax.axvline(threshold, label=f"Schwellenwert: {threshold}", color=fapsc.black, linewidth=2)

    ax.set_xlabel("Rekonstruktionsfehler in MAE", fontsize=16)
    ax.set_ylabel("Anzahl der Fehler", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)

    #if error_type=="mae":
    #    plt.title("Verteilung der Test-Rekonstruktionsfehler in MAE", fontsize=16)
    #elif error_type=="mse":
    #    plt.title("Verteilung der Test-Rekonstruktionsfehler in MAE", fontsize=16)
    plt.legend(loc='upper right', fontsize=16)
    plt.show()



def plot_calc_cm_binary(y_true, y_pred, figsize=(6,6), dpi=80):
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cm = confusion_matrix(y_true, y_pred)
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
    cmd.plot(ax=ax, colorbar=False, cmap="Greens")

    plt.xticks(ticks=[0,1], labels=["0_normal", "1_fehlerhaft"])
    plt.yticks(ticks=[0,1])
    plt.tick_params(axis='both', labelsize=16)
    
    plt.xlabel("Vorhergesagte Klasse", fontsize=16)
    plt.ylabel("Wahre Klasse", fontsize=16)
    plt.rcParams['font.size'] = 20
    #fig.tight_layout(pad=2)
    plt.show()
    
    print(classification_report(y_true, y_pred))


def plot_calc_cm(y_true, y_pred, label, dpi=80):
    
    fig, ax = plt.subplots(figsize=(7,7), dpi=dpi)
    cm = confusion_matrix(y_true, y_pred)
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
    cmd.plot(ax=ax, colorbar=False, cmap="Greens")

    plt.xticks(ticks=[x for x in range(len(label))])
    plt.yticks(ticks=[x for x in range(len(label))], labels=label)
    plt.tick_params(axis='both', labelsize=15)
    
    plt.xlabel("Vorhergesagte Klasse", fontsize=16)
    plt.ylabel("Wahre Klasse", fontsize=16)
    plt.show()
    
    print(classification_report(y_true, y_pred))



def prepare_train_test(model, x_train, x_test, max_train, len_curve, scaled=None):
    
    if scaled == True:
        reconstructed_train = model.predict(x_train)
        reconstructed_train_inverse = reconstructed_train.reshape(-1, len_curve) * max_train
        
        reconstructed_test = model.predict(x_test)
        reconstructed_test_inverse = reconstructed_test.reshape(-1, len_curve) * max_train
        
        print(f"reconstructed_train_inverse shape: {reconstructed_train_inverse.shape}")
        print(f"reconstructed_test_inverse shape: {reconstructed_test_inverse.shape}")
        
        return reconstructed_train_inverse, reconstructed_test_inverse
        
    if scaled == False:
        reconstructed_train = model.predict(x_train)
        reconstructed_train = reconstructed_train.reshape(-1, len_curve)
        
        reconstructed_test = model.predict(x_test)
        reconstructed_test = reconstructed_test.reshape(-1, len_curve)
        
        print(reconstructed_train.shape)
        print(reconstructed_test.shape)
        
        return reconstructed_train, reconstructed_test



def reconstruction_loss_mae(x_train, recon_train, x_test, recon_test):
    
    train_mae = np.mean(np.abs(x_train - recon_train), axis=1)
    test_mae = np.mean(np.abs(x_test - recon_test), axis=1)   
    
    return train_mae, test_mae, 


def reconstruction_loss_mse(x_train, recon_train, x_test, recon_test):

    train_mse = np.mean(np.square(x_train - recon_train), axis=1)
    test_mse = np.mean(np.square(x_test - recon_test), axis=1)      

    return train_mse, test_mse


def reconstruction_loss_rmse(x_train, recon_train, x_test, recon_test):

    train_rmse = np.sqrt(np.mean((x_train - recon_train)**2, axis=1))
    test_rmse = np.sqrt(np.mean((x_test - recon_test)**2, axis=1))

    return train_rmse, test_rmse



def get_predictions(y_test_binary, threshold, loss):

    anomaly_mask = pd.Series(loss) > threshold
    anomaly_prediction = np.array(anomaly_mask.map(lambda x: 1.0 if x == True else 0)).astype(int)  # anomaly=1, ok-process=0
    
    #print(anomaly_prediction)
    cm = confusion_matrix(y_test_binary, anomaly_prediction)
    tp = cm[0][0]
    fn = cm[0][1]
    fp = cm[1][0]
    tn = cm[1][1]
       
    tpr = tp/(tp+fn)
    fpr = fp/(fp+tn)
    f1 = f1_score(y_test_binary, anomaly_prediction)
    acc = accuracy_score(y_test_binary, anomaly_prediction)
     
    return tpr, fpr, f1, acc


def get_anomaly_pred_acc(y_test_binary, threshold, loss):

    anomaly_mask = pd.Series(loss) > threshold
    anomaly_prediction = np.array(anomaly_mask.map(lambda x: 1.0 if x == True else 0)).astype(int)  # anomaly=1, ok-process=0
    
    #print(anomaly_prediction)
    acc = accuracy_score(y_test_binary, anomaly_prediction)
    rec = recall_score(y_test_binary, anomaly_prediction)
    f1 = f1_score(y_test_binary, anomaly_prediction)
    print(f"Accuracy: {acc}")
    print(f"Recall: {rec}" )
    print(f"F1 Score: {f1}")

    return anomaly_prediction