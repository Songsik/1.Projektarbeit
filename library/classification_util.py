import pandas as pd
import numpy as np
import os
import json
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, cross_validate
from tensorflow.keras.models import Sequential, clone_model
from tensorflow.keras.layers import Dense, Conv1D, Flatten, MaxPooling1D, Dropout, Input, BatchNormalization
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
import pickle
from tqdm.notebook import tqdm_notebook as tq
from warnings import filterwarnings
filterwarnings('ignore')
import importlib
from library import fapsc
from library import etl_data as etl


def plot_calc_cm(y_true, y_pred, class_name, dpi=100):
    
    fig, ax = plt.subplots(figsize=(7,7), dpi=dpi)
    cm = confusion_matrix(y_true, y_pred)
    cmd = ConfusionMatrixDisplay(confusion_matrix=cm)
    cmd.plot(ax=ax, colorbar=False, cmap="Greens")

    plt.xticks(ticks=[x for x in range(len(class_name))])
    plt.yticks(ticks=[x for x in range(len(class_name))], labels=class_name)
    plt.tick_params(axis='both', labelsize=15)
    
    plt.xlabel("Vorhergesagte Klasse", fontsize=15)
    plt.ylabel("Wahre Klasse", fontsize=15)
    plt.rcParams['font.size'] = 15
    plt.show()
    
    print(classification_report(y_true, y_pred))



def plot_loss_acc(history):

    plot_loss(history)
    plot_acc(history)


def plot_loss(history):

    plt.figure(figsize=(10,7))
    plt.title('Loss', fontsize=30)
    plt.plot(history.history['loss'], label='train loss', linewidth=3)
    plt.plot(history.history['val_loss'], label='val loss', linewidth=3)
    plt.legend(loc='upper right', fontsize=20)
    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)
    plt.show()



def plot_acc(history):

    plt.figure(figsize=(10,7))
    plt.title('Accuracy', fontsize=30)
    plt.plot(history.history['accuracy'], label='train acc', linewidth=3)
    plt.plot(history.history['val_accuracy'], label='val acc', linewidth=3)
    plt.legend(loc='lower right', fontsize=20)
    plt.tick_params(axis="x", labelsize=16)
    plt.tick_params(axis="y", labelsize=16)
    plt.show()



def plot_loss_acc_parallel(history, figsize=(20,7), dpi=80):

    fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    ax[0].set_title('Loss', fontsize=30)
    ax[0].plot(history['loss'], label='train loss', linewidth=4)
    ax[0].plot(history['val_loss'], label='val loss', linewidth=4)
    ax[0].legend(loc='upper right', fontsize=24)
    ax[0].tick_params(axis="x", labelsize=20)
    ax[0].tick_params(axis="y", labelsize=20)
    ax[0].set_xlabel("epochs", fontsize=24)

    ax[1].set_title('Accuracy', fontsize=30)
    ax[1].plot(history['accuracy'], label='train acc', linewidth=4)
    ax[1].plot(history['val_accuracy'], label='val acc', linewidth=4)
    ax[1].legend(loc='lower right', fontsize=24)
    ax[1].tick_params(axis="x", labelsize=20)
    ax[1].tick_params(axis="y", labelsize=20)
    ax[1].set_xlabel("epochs", fontsize=24)

    fig.tight_layout(pad=3)
    plt.show()



def find_wrong_classification(y_true, y_pred, x_test, df_list):
    
    misclassified = {"true_label_0": [], "true_label_1": [], "true_label_2": [],
                     "true_label_3": [], "true_label_4": [], "true_label_5": []}
    
    for num, val in enumerate(zip(y_true, y_pred)):
        if val[0]!=val[1]:
            misclassified[f"true_label_{val[0]}"].append(num)
            
    df_x_test = pd.DataFrame(x_test.reshape(-1, x_test.shape[1]).transpose())
    df_miss = pd.DataFrame()

    for key in misclassified:
        df_miss = pd.concat([df_miss, df_x_test[misclassified[key]]], axis=1)

    misclassified_curve_index_each_df = find_curve_index_in_each_df(misclassified, df_miss, df_list)

    return misclassified_curve_index_each_df, misclassified, df_miss, df_x_test



def find_curve_index_in_each_df(misclassified, df_miss, df_list):

    misclassified_curve_index_each_df = {"df0": [], "df1": [], "df2": [],
                                         "df3": [], "df4": [], "df5": []}

    for num, key in enumerate(misclassified):
        if misclassified[key]==[]:
            pass
        
        elif (misclassified[key]!=[]) & (num==0):
            for col1 in df_miss:
                for col2 in df_list[0]:
                    if round(df_miss.iloc[:len(df_list[0])][col1], 4).equals(round(df_list[0][col2], 4)):
                        misclassified_curve_index_each_df["df0"].append(col2)
                        
        elif (misclassified[key]!=[]) & (num==1):
            for col1 in df_miss:
                for col2 in df_list[1]:
                    if round(df_miss.iloc[:len(df_list[1])][col1], 4).equals(round(df_list[1][col2], 4)):
                        misclassified_curve_index_each_df["df1"].append(col2)
            
        elif (misclassified[key]!=[]) & (num==2):
            for col1 in df_miss:
                for col2 in df_list[2]:
                    if round(df_miss.iloc[:len(df_list[2])][col1], 4).equals(round(df_list[2][col2], 4)):
                        misclassified_curve_index_each_df["df2"].append(col2)
                        
        elif (misclassified[key]!=[]) & (num==3):
            for col1 in df_miss:
                for col2 in df_list[3]:
                    if round(df_miss.iloc[:len(df_list[3])][col1], 4).equals(round(df_list[3][col2], 4)):
                        misclassified_curve_index_each_df["df3"].append(col2)
                        
        elif (misclassified[key]!=[]) & (num==4):
            for col1 in df_miss:
                for col2 in df_list[4]:
                    if round(df_miss.iloc[:len(df_list[4])][col1], 4).equals(round(df_list[4][col2], 4)):
                        misclassified_curve_index_each_df["df4"].append(col2)
                        
        elif (misclassified[key]!=[]) & (num==5):
            for col1 in df_miss:
                for col2 in df_list[5]:
                    if round(df_miss.iloc[:len(df_list[5])][col1], 4).equals(round(df_list[5][col2], 4)):
                        misclassified_curve_index_each_df["df5"].append(col2)
                    
    return misclassified_curve_index_each_df
    
        


def final_evaluation(testing_model, feature, label, num_trials, epochs, batch_size, param_grid):

    result_dict = {"acc": [],
                   "rec": [],
                   "f1" : []}

    for i in tq(range(num_trials)):
        print(f"Start {i}ter Lauf=========================================================================================")

        xtrain, xtest, ytrain, ytest = train_test_split(feature, label, random_state=i, test_size=0.25, shuffle=True)
        feature_shuffle = np.concatenate((xtrain, xtest))
        label_shuffle = np.concatenate((ytrain, ytest))
        
        cv_outer = KFold(n_splits=4, shuffle=True, random_state=1)

        for train, test in cv_outer.split(label_shuffle):
            x_train, x_test = feature_shuffle[train, :], feature_shuffle[test, :]
            y_train, y_test = label_shuffle[train], label_shuffle[test]

            cv_inner = KFold(n_splits=4, shuffle=True, random_state=1)

            model = KerasClassifier(build_fn=testing_model, epochs=epochs, batch_size=batch_size, verbose=1)

            clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv_inner, refit=True)
            result = clf.fit(x_train, y_train)
            best_model = result.best_estimator_

            y_true = np.argmax(y_test, axis=1)
            y_pred = best_model.predict(x_test)

            result_dict['acc'].append(accuracy_score(y_true, y_pred))
            result_dict['rec'].append(recall_score(y_true, y_pred, average='weighted'))
            result_dict['f1'].append(f1_score(y_true, y_pred, average='weighted'))
                
        print(f"Ende {i}ter Lauf=========================================================================================\n")


    return result_dict



def box_plot_color(ax, data, edge_color, fill_color):
    bp = ax.boxplot(data, patch_artist=True, showmeans=True,
                    meanprops={"marker":"o",
                               "markerfacecolor":"white", 
                               "markeredgecolor":fapsc.black,
                               "markersize":"10"})
    
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    for patch in bp['boxes']:
        patch.set(facecolor=fill_color)       
        
    return bp


def boxplot_model_results(result_dict, title, xlist, color, size=(8, 6), dpi=80, lower=None, upper=None):

    #data = [result_dict["acc"], result_dict["rec"], result_dict["f1"]]
    #data_name = ["Accuracy", "Weighted_Recall", "Weighted_F1"]
    data = []
    for key in result_dict:
        data.append(result_dict[key])
    data_name = xlist

    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    bp = box_plot_color(ax, data, fapsc.black, color)

    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=15)
    ax.set_ylabel('Ergebnisse der Testdaten', fontsize=15)
    ax.set_ylim(lower, upper)
    ax.set_xticklabels(data_name)


    ax.yaxis.grid()
    ax.tick_params(axis='both', labelsize=15)

    plt.show()




def cnn_structure(conv, maxpool, bn_conv, dropout_conv, dense, bn_dense, dropout_dense):
    
    model = Sequential()
    
    if conv >= 1:
        model.add(Conv1D(32, kernel_size=8, activation="relu", input_shape=(920, 1)))
        if maxpool[0]:
            model.add(MaxPooling1D(3))
        if bn_conv[0]:
            model.add(BatchNormalization())
        if dropout_conv[0]:
            model.add(Dropout(0.1))
        
    if conv >= 2:
        model.add(Conv1D(32, kernel_size=8, activation="relu"))
        if maxpool[1]:
            model.add(MaxPooling1D(3))
        if bn_conv[1]:
            model.add(BatchNormalization())
        if dropout_conv[1]:
            model.add(Dropout(0.1))
    
    if conv >= 3:
        model.add(Conv1D(32, kernel_size=8, activation="relu"))
        if maxpool[2]:
            model.add(MaxPooling1D(3))
        if bn_conv[2]:
            model.add(BatchNormalization())
        if dropout_conv[2]:
            model.add(Dropout(0.1))
            
    if conv >= 4:
        model.add(Conv1D(32, kernel_size=8, activation="relu"))
        if maxpool[3]:
            model.add(MaxPooling1D(3))
        if bn_conv[3]:
            model.add(BatchNormalization())
        if dropout_conv[3]:
            model.add(Dropout(0.1))

    
    model.add(Flatten())
    
    if dense >= 1:
        model.add(Dense(80, activation="relu"))
        if bn_dense[0]:
            model.add(BatchNormalization())
        if dropout_dense[0]:
            model.add(Dropout(0.1))
            
    if dense >= 2:
        model.add(Dense(80, activation="relu"))
        if bn_dense[1]:
            model.add(BatchNormalization())
        if dropout_dense[1]:
            model.add(Dropout(0.1))
    
    if dense >= 3:
        model.add(Dense(80, activation="relu"))
        if bn_dense[2]:
            model.add(BatchNormalization())
        if dropout_dense[2]:
            model.add(Dropout(0.1))
            
    if dense >= 4:
        model.add(Dense(80, activation="relu"))
        if bn_dense[3]:
            model.add(BatchNormalization())
        if dropout_dense[3]:
            model.add(Dropout(0.1))
            
    model.add(Dense(8, activation="softmax"))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print(model.summary())
    return model


def cnn_structure_optimization(model_dict, x_train, y_train, epochs, batch_size):

    model = KerasClassifier(build_fn=cnn_structure, epochs=epochs, batch_size=batch_size)

    clf = GridSearchCV(estimator=model, param_grid=model_dict, cv=5, refit=True, verbose=10) #scoring_default -> accuracy
    result = clf.fit(x_train, y_train) 

    # save model structure and cv results as mean test score in acc
    if os.path.exists("results/cnn_structure_optimization_v4.1.json") == False:
        with open("results/cnn_structure_optimization_v4.1.json", "w") as f:
            json.dump([model_dict, result.cv_results_["mean_test_score"][0]], f)
            f.close()
    else: 
        with open("results/cnn_structure_optimization_v4.1.json", "r") as f:
            data = json.load(f)
            data.append(model_dict)
            data.append(result.cv_results_["mean_test_score"][0])

        with open("results/cnn_structure_optimization_v4.1.json", "w") as f:
            json.dump(data, f)
            f.close()

    print(result.cv_results_["mean_test_score"][0])
    #return pd.DataFrame(result.cv_results_)



def grouped_barplot(leftlist, leftlabel, leftcolor, rightlist, rightlabel, rightcolor, title, xtickslist, size=(8,6), dpi=100):

    labels = xtickslist
    left = leftlist
    right = rightlist

    x = np.arange(len(labels))  # the label locations
    width = 0.4  # the width of the bars

    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    rects1 = ax.bar(x - width/2, left, width, color=leftcolor, label=leftlabel)
    ax.bar_label(rects1, padding=3, fontsize=14)

    rects2 = ax.bar(x + width/2, right, width, color=rightcolor, label=rightlabel)
    ax.bar_label(rects2, padding=3, fontsize=14)

    ax.set_title(title, fontsize=18)
    ax.set_ylabel(f"Mean 5 Fold Cross\n Validation Accuracy", fontsize=16)
    ax.set_xticks(x, labels, fontsize=16)
    ax.set_ylim(0, 1.25)

    plt.legend(loc='upper right', fontsize=14)

    plt.show()



def single_barplot(values, valuecolor, title, xtickslist, size=(8,6), width=0.4, rotation=45, dpi=100, ylim=1.25):

    labels = xtickslist

    x_position = np.arange(len(labels))  # the label, bar locations
    
    fig, ax = plt.subplots(figsize=size, dpi=dpi)
    rects = ax.bar(x_position, values, width, color=valuecolor)
    ax.bar_label(rects, padding=3, fontsize=12)

    ax.set_title(title, fontsize=16)
    ax.set_ylabel(f"Mean 5 Fold Cross\n Validation Accuracy", fontsize=14)
    ax.set_xticks(x_position, labels, fontsize=12, rotation=rotation)
    ax.set_ylim(0, ylim)

    #plt.legend(loc='upper right', fontsize=12)

    plt.show()



def final_evaluation_simple(testing_model, feature, label, num_trials, epochs, batch_size, classes, optimized=None):

    result_dict = {"acc": [],
                   "rec": [],
                   "pre": [],
                   "f1" : []}

    for i in tq(range(num_trials)):
        print(f"Start {i}ter Lauf=========================================================================================")

        x_train, x_test, y_train, y_test = train_test_split(feature, label, random_state=i, test_size=0.25, shuffle=True)
        x_train = x_train.reshape(-1, feature.shape[1], 1)
        x_test = x_test.reshape(-1, feature.shape[1], 1)

        tf.keras.backend.clear_session()

        model = clone_model(testing_model)
        model.compile(optimizer=Adam(learning_rate=0.0005), loss="categorical_crossentropy", metrics=["accuracy"])
        
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
        print(model.evaluate(x_test, y_test, batch_size=batch_size))
        model.save(f"cnn_model_v4.1/cnn_model_{i}.h5")
        
        y_pred = np.argmax(model.predict(x_test), axis=1)
        y_true = np.argmax(y_test, axis=1)  
        
        result_dict['acc'].append(accuracy_score(y_true, y_pred))
        result_dict['rec'].append(recall_score(y_true, y_pred, average='macro'))
        result_dict['pre'].append(precision_score(y_true, y_pred, average='macro'))
        result_dict['f1'].append(f1_score(y_true, y_pred, average='macro'))
                
        print(f"Ende {i}ter Lauf=========================================================================================\n")
    
    return result_dict




def find_misclassification_with_duplicates(y_test, y_pred, df_list, wrong_pred_dict, x_test_df, rounding=4):
    
    for num, key in enumerate(wrong_pred_dict):
        for col1, val in enumerate(zip(y_test, y_pred)):
            if num==0:
                for col2 in df_list[num]:
                    if (val[0]==num) and (val[0] != val[1]) and (round(x_test_df.iloc[:len(df_list[num])][col1], rounding).equals(round(df_list[num][col2], rounding))==True):
                        wrong_pred_dict[key]["df_col"].append(col2)
                        wrong_pred_dict[key]["true"].append(val[0])
                        wrong_pred_dict[key]["misclassified_as"].append(val[1])
                            
            elif num==1:
                for col2 in df_list[num]:
                    if (val[0]==num) and (val[0] != val[1]) and (round(x_test_df.iloc[:920][col1], rounding).equals(round(df_list[num].iloc[:920][col2], rounding))==True):
                        wrong_pred_dict[key]["df_col"].append(col2)
                        wrong_pred_dict[key]["true"].append(val[0])
                        wrong_pred_dict[key]["misclassified_as"].append(val[1])
                            
            elif num==2:
                for col2 in df_list[num]:
                    if (val[0]==num) and (val[0] != val[1]) and (round(x_test_df.iloc[:len(df_list[num])][col1], rounding).equals(round(df_list[num][col2], rounding))==True):
                        wrong_pred_dict[key]["df_col"].append(col2)
                        wrong_pred_dict[key]["true"].append(val[0])
                        wrong_pred_dict[key]["misclassified_as"].append(val[1])
                            
            elif num==3:
                for col2 in df_list[num]:
                    if (val[0]==num) and (val[0] != val[1]) and (round(x_test_df.iloc[:len(df_list[num])][col1], rounding).equals(round(df_list[num][col2], rounding))==True):
                        wrong_pred_dict[key]["df_col"].append(col2)
                        wrong_pred_dict[key]["true"].append(val[0])
                        wrong_pred_dict[key]["misclassified_as"].append(val[1])
                            
            elif num==4:
                for col2 in df_list[num]:
                    if (val[0]==num) and (val[0] != val[1]) and (round(x_test_df.iloc[:len(df_list[num])][col1], rounding).equals(round(df_list[num][col2], rounding))==True):
                        wrong_pred_dict[key]["df_col"].append(col2)
                        wrong_pred_dict[key]["true"].append(val[0])
                        wrong_pred_dict[key]["misclassified_as"].append(val[1])
                            
            elif num==5:
                for col2 in df_list[num]:
                    if (val[0]==num) and (val[0] != val[1]) and (round(x_test_df.iloc[:len(df_list[num])][col1], rounding).equals(round(df_list[num][col2], rounding))==True):
                        wrong_pred_dict[key]["df_col"].append(col2)
                        wrong_pred_dict[key]["true"].append(val[0])
                        wrong_pred_dict[key]["misclassified_as"].append(val[1])

            elif num==6:
                for col2 in df_list[num]:
                    if (val[0]==num) and (val[0] != val[1]) and (round(x_test_df.iloc[:len(df_list[num])][col1], rounding).equals(round(df_list[num][col2], rounding))==True):
                        wrong_pred_dict[key]["df_col"].append(col2)
                        wrong_pred_dict[key]["true"].append(val[0])
                        wrong_pred_dict[key]["misclassified_as"].append(val[1])

            elif num==7:
                for col2 in df_list[num]:
                    if (val[0]==num) and (val[0] != val[1]) and (round(x_test_df.iloc[:len(df_list[num])][col1], rounding).equals(round(df_list[num][col2], rounding))==True):
                        wrong_pred_dict[key]["df_col"].append(col2)
                        wrong_pred_dict[key]["true"].append(val[0])
                        wrong_pred_dict[key]["misclassified_as"].append(val[1])
    
    return wrong_pred_dict



def plot_wrong_preds(dfa, dfwrong, colora, labela, dfb, colorb, labelb, figsize=(8,4), dpi=80):

    fig, ax = plt.subplots(1, 2, figsize=figsize, dpi=dpi)

    ax[0].plot(etl.set_time(dfa), dfa[dfwrong].values, color=colora, linewidth=3)
    ax[0].plot([], [], color=colora, label=labela)
    ax[0].tick_params(axis='both', labelsize=14)
    ax[0].legend(loc="upper left", fontsize=16)
    ax[0].grid()
    ax[0].yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))
    ax[0].xaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))

    ax[1].plot(etl.set_time(dfb), dfb.values, color=colorb, linewidth=3)
    ax[1].plot([], [], color=colorb, label=labelb)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[1].legend(loc="upper left", fontsize=16)
    ax[1].grid()
    ax[1].yaxis.set_major_formatter(StrMethodFormatter('{x:,.1f}'))

    plt.tight_layout(pad=1)
    plt.show()