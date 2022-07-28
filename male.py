# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 22:20:21 2022

@author: Akyra
"""

# separating female data 
male_df = jpn_df[["Year", "Age", "Male"]]

male_df = male_df.pivot(index= "Year", columns = "Age", values="Male")

sns.heatmap(male_df)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
male_df = scaler.fit_transform(male_df)

MnumTrain = int(0.6 * len(male_df))
MnumVal = int(0.2 * len(male_df))
MnumTest = len(male_df) - numTrain - numVal
print(f"# Train: {MnumTrain}, # Val: {MnumVal}, # Test: {MnumTest}")

# Num. of input time series.
MnumTS = male_df.shape[1]

# How many prev. months to use.
seqLength = 10

# Predict the next month ahead.
ahead = 1

# The index of the first target.
delay = (seqLength+ahead-1)


MtrainDS = \
  timeseries_dataset_from_array(
    male_df[:-delay],
    targets=male_df[delay:],
    sequence_length=seqLength,
    end_index=MnumTrain)

MvalDS = \
  timeseries_dataset_from_array(
    male_df[:-delay],
    targets=male_df[delay:],
    sequence_length=seqLength,
    start_index=MnumTrain,
    end_index=MnumTrain+MnumVal)
  
MtestDS = \
  timeseries_dataset_from_array(
    male_df[:-delay],
    targets=male_df[delay:],
    sequence_length=seqLength,
    start_index=MnumTrain+MnumVal) 

MX_train = np.concatenate(list(MtrainDS.map(lambda x, y: x)))
MX_train.shape

MY_train = np.concatenate(list(MtrainDS.map(lambda x, y: y)))
MY_train.shape

MY_train = np.concatenate(list(MtrainDS.map(lambda x, y: y)))
MY_val = np.concatenate(list(MvalDS.map(lambda x, y: y)))
MY_test = np.concatenate(list(MtestDS.map(lambda x, y: y)))


# LSTM ------------------------------------------------------------------------------------------------
tf.random.set_seed(1)

modelLSTM = Sequential([
    LSTM(32, input_shape=(seqLength, numTS), return_sequences=True),
    Dropout(0.2), 
    LSTM(32, activation="relu"), 
    Dropout(0.2), 
    Dense(1, activation="relu")
])

modelLSTM.compile(loss="mse", optimizer="adam")

es = EarlyStopping(patience=50, restore_best_weights=True, verbose=1)

%time hist = modelLSTM.fit(MtrainDS, epochs=1_000, validation_data=MvalDS, callbacks=[es], verbose=1);

modelLSTM.evaluate(MvalDS, verbose=1)
modelLSTM.evaluate(MtrainDS, verbose=1)
modelLSTM.evaluate(MtestDS, verbose=1)


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

%time hist = modelCNN.fit(MtrainDS, epochs=1_000, validation_data=MvalDS, callbacks=[es], verbose=1)

modelCNN.evaluate(MvalDS, verbose=1)


 