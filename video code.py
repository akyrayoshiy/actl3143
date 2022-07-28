# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 22:39:29 2022

@author: Akyra
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import seaborn as sns


import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

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

train_f = female_df.loc[female_df["Year"] <= 1999]
val_f = female_df.loc[(female_df["Year"]>1999) & (female_df["Year"]<=2009)]
test_f = female_df.loc[female_df["Year"] >2009]

numTrain = int(len(train_f))
numVal = int(len(val_f))
numTest = int(len(test_f))
print(f"# Train: {numTrain}, # Val: {numVal}, # Test: {numTest}")


pip install --upgrade tensorflow
from tensorflow.keras.utils import timeseries_dataset_from_array
# Num. of input time series.
numTS = female_df.shape[1]

# How many prev. months to use.
seqLength = 10

# Predict the next month ahead.
ahead = 1

# The index of the first target.
delay = (seqLength+ahead-1)

# Which suburb to predict.
target = female_df["Female"]

trainDS = \
  timeseries_dataset_from_array(
    female_df[:-delay],
    targets=target[delay:],
    sequence_length=seqLength,
    end_index=numTrain)

valDS = \
  timeseries_dataset_from_array(
    female_df[:-delay],
    targets=target[delay:],
    sequence_length=seqLength,
    start_index=numTrain,
    end_index=numTrain+numVal)
  
testDS = \
  timeseries_dataset_from_array(
    female_df[:-delay],
    targets=target[delay:],
    sequence_length=seqLength,
    start_index=numTrain+numVal) 

X_train = np.concatenate(list(trainDS.map(lambda x, y: x)))
y_train = np.concatenate(list(trainDS.map(lambda x, y: y)))

X_train.shape
y_train.shape

y_train = np.concatenate(list(trainDS.map(lambda x, y: y)))
y_val = np.concatenate(list(valDS.map(lambda x, y: y)))
y_test = np.concatenate(list(testDS.map(lambda x, y: y)))


from tensorflow.keras.layers import Input, Flatten
tf.random.set_seed(1)
modelDense = Sequential([
    Input(shape=(seqLength, numTS)),
    Flatten(),
    Dense(50, activation="leaky_relu"),
    Dense(20, activation="leaky_relu"),
    Dense(1, activation="linear")
])
modelDense.compile(loss="mse", optimizer="adam")
print(f"This model has {modelDense.count_params()} parameters.")

es = EarlyStopping(patience=50, restore_best_weights=True, verbose=1)
%time hist = modelDense.fit(trainDS, epochs=1_000, validation_data=valDS, callbacks=[es], verbose=1)

modelDense.evaluate(valDS, verbose=1)

p = modelDense.predict(X_train)


from tensorflow.keras.layers import SimpleRNN

tf.random.set_seed(1)

modelSimple = Sequential([
    SimpleRNN(50, input_shape=(seqLength, numTS)),
    Dense(1, activation="linear")
])
modelSimple.compile(loss="mse", optimizer="adam")
print(f"This model has {modelSimple.count_params()} parameters.")

es = EarlyStopping(patience=50, restore_best_weights=True, verbose=1)
%time hist = modelSimple.fit(trainDS, epochs=1_000, validation_data=valDS, callbacks=[es], verbose=1);

from tensorflow.keras.layers import LSTM

tf.random.set_seed(1)

modelLSTM = Sequential([
    LSTM(50, input_shape=(seqLength, numTS)),
    Dense(1, activation="linear")
])

modelLSTM.compile(loss="mse", optimizer="adam")

es = EarlyStopping(patience=50, restore_best_weights=True, verbose=1)

%time hist = modelLSTM.fit(trainDS, epochs=1_000, validation_data=valDS, callbacks=[es], verbose=1);


p = modelLSTM.predict(X_train)




from tensorflow.keras.layers import GRU

tf.random.set_seed(1)

modelGRU = Sequential([
    GRU(50, input_shape=(seqLength, numTS)),
    Dense(1, activation="linear")
])

modelGRU.compile(loss="mse", optimizer="adam")

es = EarlyStopping(patience=50, restore_best_weights=True, verbose=1)

%time hist = modelGRU.fit(trainDS, epochs=1_000, validation_data=valDS, callbacks=[es], verbose=1)



# GRU ----------------------------------------------------------------------------------------------------
tf.random.set_seed(1)

modelGRU = Sequential([
    GRU(50, input_shape=(seqLength, numTS)),
    Dense(numTS, activation="linear")
])

modelGRU.compile(loss="mse", optimizer="adam")

es = EarlyStopping(patience=50, restore_best_weights=True, verbose=1)

%time hist = modelGRU.fit(trainDS, epochs=1_000, validation_data=valDS, callbacks=[es], verbose=1)





# Benchmark model - using prediction rule 

val_female_data = female_df.loc[(female_df["Year"] >1990) & (female_df["Year"]<2005) ]

scaled = scaler.fit_transform(val_female_data)
val_y_female_data = scaled[:, 2]

val_female_pred = female_df.loc[(female_df["Year"] >1989) & (female_df["Year"]<2004)]
scaled2 = scaler.fit_transform(val_female_pred)
val_y_female_pred = scaled2[:,2]

BL_mseF = mean_squared_error(val_y_female_data, val_y_female_pred)
BL_mseF


val_male_data = male_df.loc[(male_df["Year"] >1990) & (male_df["Year"]<2005) ]

scaled3 = scaler.fit_transform(val_male_data)
val_y_male_data = scaled3[:, 2]

val_male_pred = male_df.loc[(male_df["Year"] >1989) & (male_df["Year"]<2004)]
scaled4 = scaler.fit_transform(val_male_pred)
val_y_male_pred = scaled4[:,2]

BL_mseM = mean_squared_error(val_y_male_data, val_y_male_pred)
BL_mseM

