# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 00:06:16 2022

@author: Akyra
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import timeseries_dataset_from_array

%load_ext watermark
%watermark -p numpy,pandas,tensorflow

jpn_df = pd.read_fwf("C:/Users/Akyra/Desktop/JPN/STATS/Mx_1x1.txt", skiprows=[0,1])

jpn_df = jpn_df.drop(jpn_df.index[jpn_df["Age"] =="00"])
jpn_df = jpn_df.drop(jpn_df.index[jpn_df["Age"] =="01"])
jpn_df = jpn_df.drop(jpn_df.index[jpn_df["Age"] =="02"])
jpn_df = jpn_df.drop(jpn_df.index[jpn_df["Age"] =="03"])
jpn_df = jpn_df.drop(jpn_df.index[jpn_df["Age"] =="04"])
jpn_df = jpn_df.drop(jpn_df.index[jpn_df["Age"] =="05"])
jpn_df = jpn_df.drop(jpn_df.index[jpn_df["Age"] =="06"])
jpn_df = jpn_df.drop(jpn_df.index[jpn_df["Age"] =="07"])
jpn_df = jpn_df.drop(jpn_df.index[jpn_df["Age"] =="08"])
jpn_df = jpn_df.drop(jpn_df.index[jpn_df["Age"] =="09"])
jpn_df = jpn_df.drop(jpn_df.index[jpn_df["Age"] =="10+"])

jpn_df.head(10)

jpn_df.info()
jpn_df.describe()

sns.scatterplot(x="Age", y="Female", data = jpn_df)
sns.scatterplot(x="Age", y="Male", data = jpn_df)


# converting all columns to numeric 
jpn_df["Age"] = pd.to_numeric(jpn_df["Age"])
jpn_df["Female"] = pd.to_numeric(jpn_df["Female"])
jpn_df["Male"] = pd.to_numeric(jpn_df["Male"])
jpn_df["Total"] = pd.to_numeric(jpn_df["Total"])

# separating female data 
female_df = jpn_df[["Year", "Age", "Female"]]

female_df = female_df.pivot(index= "Year", columns = "Age", values="Female")

sns.heatmap(female_df)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
female_df = scaler.fit_transform(female_df)

numTrain = int(0.6 * len(female_df))
numVal = int(0.2 * len(female_df))
numTest = len(female_df) - numTrain - numVal
print(f"# Train: {numTrain}, # Val: {numVal}, # Test: {numTest}")

# Num. of input time series.
numTS = female_df.shape[1]

# How many prev. months to use.
seqLength = 10

# Predict the next month ahead.
ahead = 1

# The index of the first target.
delay = (seqLength+ahead-1)


trainDS = \
  timeseries_dataset_from_array(
    female_df[:-delay],
    targets=female_df[delay:],
    sequence_length=seqLength,
    end_index=numTrain)

valDS = \
  timeseries_dataset_from_array(
    female_df[:-delay],
    targets=female_df[delay:],
    sequence_length=seqLength,
    start_index=numTrain,
    end_index=numTrain+numVal)
  
testDS = \
  timeseries_dataset_from_array(
    female_df[:-delay],
    targets=female_df[delay:],
    sequence_length=seqLength,
    start_index=numTrain+numVal) 

X_train = np.concatenate(list(trainDS.map(lambda x, y: x)))
X_train.shape

Y_train = np.concatenate(list(trainDS.map(lambda x, y: y)))
Y_train.shape

Y_train = np.concatenate(list(trainDS.map(lambda x, y: y)))
Y_val = np.concatenate(list(valDS.map(lambda x, y: y)))
Y_test = np.concatenate(list(testDS.map(lambda x, y: y)))


# LSTM ------------------------------------------------------------------------------------------------
tf.random.set_seed(1)

modelLSTM = Sequential([
    LSTM(128, input_shape=(seqLength, numTS), return_sequences=True, name="LSTM_1"),
    Dropout(0.2, name="Dropout_1"), 
    LSTM(128, activation="relu", name="LSTM_2"), 
    Dropout(0.2, name="Dropout_2"), 
    Dense(1, activation="relu", name="output")
])

modelLSTM.compile(loss="mse", optimizer="adam")

es = EarlyStopping(patience=50, restore_best_weights=True, verbose=1)

%time hist = modelLSTM.fit(trainDS, epochs=1_000, validation_data=valDS, callbacks=[es], verbose=1);


modelLSTM.evaluate(valDS, verbose=1)
modelLSTM.evaluate(testDS, verbose=1)
modelLSTM.evaluate(trainDS, verbose=1)

plot_model(modelLSTM, show_shapes=True)

pip install pydot

pip install pydotplus
pip install graphviz

conda install graphviz
conda install pydot
conda install pydotplus

modelLSTM.summary()


# CNN -------------------------------------------------------------------------------------------------------

from tensorflow.keras.layers \
  import Rescaling, Conv2D, MaxPooling2D, Flatten


tf.random.set_seed(123)

modelCNN = Sequential([
    Input(shape=(seqLength, numTS, 1)),
  Conv2D(128, 3, padding="same", activation="relu", name="conv1"),
  MaxPooling2D(name="pool1"),
  Conv2D(64, 3, padding="same", activation="relu", name="conv2"),
  MaxPooling2D(name="pool2"),
  Flatten(), 
  Dense(128, activation="relu"),
  Dense(numTS, activation="relu"), 
])

modelCNN.compile(loss="mse", optimizer="adam")

es = EarlyStopping(patience=50, restore_best_weights=True, verbose=1)

%time hist = modelCNN.fit(trainDS, epochs=1_000, validation_data=valDS, callbacks=[es], verbose=1)

modelCNN.evaluate(valDS, verbose=1)
modelCNN.evaluate(trainDS, verbose=1)

plot_model(modelCNN, show_shapes=True)

modelCNN.summary()



 






