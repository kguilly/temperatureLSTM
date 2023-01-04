from __future__ import print_function
import argparse
import sys
import numpy as np
ID = np.random.randint(2) # Randomly generate  0 and 1

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #str(ID) # Randomly select GPU

import tensorflow as tf
import keras
from keras.models import load_model
from weather import Weather
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, LSTM, Dense,  Add

from sklearn.preprocessing import MinMaxScaler
import matplotlib
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.utils.fixes import parse_version
from scipy.signal import find_peaks
from keras.utils import to_categorical

#Setting gpus by Puru
gpus = tf.config.list_physical_devices('GPU')
print("*** Checking gpus: ",gpus)
if gpus:
    try:
    # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
        print(e)
## GPU setting ends here

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 24})
# plt.rcParams['figure.figsize'] = 16, 12

if parse_version(matplotlib.__version__) >= parse_version('2.1'):
    density_param = {'density': True}
else:
    density_param = {'normed': True}



m = Weather()
m.latent_dim = 256

stationList = ['BMTN', 'CCLA', 'FARM', 'HUEY', 'LXGN','ELST','FCHV','QKSD','LSML','LGRN','CROP','DANV']  # include your station list here

# Default station and paramID
station = 'BMTN'
paramID = 10  # 1: Temperature, 2: Relative Humidity, 3: Soil temperature, 4: Wind speed, 5: Pressure, and 6: Soil moisture

train = True
###############################################
# Training date range
during = 90  # How many days of data in each year
year_during = 2  # How many years of data
start_date = "0715"  # Start date of each year
year = "2018"
seq = 5

# Test date range
T_year_during = 1
T_during = 2
T_start_date = "0820"
T_year = "2020"

mapper_epochs = 100
model_epochs = 60
decoderLen = 3 # 12 for60 minutes prediction
##############################################

print(len(sys.argv))
if len(sys.argv) > 1 and len(sys.argv) < 3:
    print("please follow the execution like: ")
    print("python modelet_name (say, kentucky_macro.py) station_name (say, BMBL) parmaID (say, 1)")
    exit(0)

# python kentucky_micro BMBL 3
if len(sys.argv) == 3:
    station = sys.argv[1]
    paramID = sys.argv[2]
    paramID = int(paramID)

if station not in stationList:
    print("Invalid station Name, please check the list")
    exit(1)

if paramID > 4 or paramID < 1:
    print("invalid paramID")
    exit(1)

m.station = station

if paramID == 1:
    m.test_feature_list = ['TAIR']
    m.feature_list_full = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'TAIR', 'THMP', 'TDPT', 'ST02', 'SM02', 'SRAD',
                           'PRES', 'WDSD', 'RELH', 'DayOfTheYear', 'WSSD']
    m.feature_list_full_value = ['TAIR', 'THMP', 'TDPT', 'ST02', 'SM02', 'SRAD', 'PRES', 'WDSD', 'RELH', 'DayOfTheYear',
                                 'WSSD']

    m.wrf_feature_list_full = ['Year', 'Month', 'Date', 'Hour', 'Minute', 'ATT66', 'ATT67', 'ATT59', 'ATT111', 'ATT73',
                               'ATT94', 'ATT28', 'ATT69', 'ATT109', 'ATT68']
    m.wrf_feature_list_value = ['ATT66', 'ATT67', 'ATT59', 'ATT111', 'ATT73', 'ATT94', 'ATT28', 'ATT69', 'ATT109',
                                'ATT68']
if paramID == 2:
    m.test_feature_list = ['RELH']
    m.feature_list_full = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'RELH','SRAD', 'WSSD', 'WDSD', 'WSMX', 'WSPD',
                           'WSMN', 'THMP', 'PRES', 'TAIR']
    m.feature_list_full_value = ['RELH','SRAD', 'WSSD', 'WDSD', 'WSMX', 'WSPD', 'WSMN', 'THMP', 'PRES', 'TAIR']

    m.wrf_feature_list_full = ['Year', 'Month', 'Date', 'Hour', 'Minute', 'ATT70', 'ATT4', 'ATT108', 'ATT110', 'ATT112',
                               'ATT91', 'ATT90', 'ATT61', 'ATT113', 'ATT111']
    m.wrf_feature_list_value = ['ATT70', 'ATT4', 'ATT108', 'ATT110', 'ATT112', 'ATT91', 'ATT90', 'ATT61', 'ATT113',
                                'ATT111']

if paramID == 3:
    m.test_feature_list = ['WSPD']
    m.feature_list_full = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'WSPD', 'WSMX', 'WSMN', 'WSSD', 'PRES', 'ST02',
                           'RELH', 'SM02']
    m.feature_list_full_value = ['WSPD', 'WSMX', 'WSMN', 'WSSD', 'PRES', 'ST02', 'RELH', 'SM02']

    m.wrf_feature_list_full = ['Year', 'Month', 'Date', 'Hour', 'Minute', 'ATT8', 'ATT73', 'ATT89', 'ATT115', 'ATT98',
                               'ATT18', 'ATT121', 'ATT125']
    m.wrf_feature_list_value = ['ATT8', 'ATT73', 'ATT89', 'ATT115', 'ATT98', 'ATT18', 'ATT121', 'ATT125']

if paramID == 4:
    m.test_feature_list = ['PRES']
    m.feature_list_full = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'PRES', 'TDPT', 'TAIR', 'THMP', 'ST02', 'WSMN',
                           'WSPD']
    m.feature_list_full_value = ['PRES', 'TDPT', 'TAIR', 'THMP', 'ST02', 'WSMN', 'WSPD']

    m.wrf_feature_list_full = ['Year', 'Month', 'Date', 'Hour', 'Minute', 'ATT57', 'ATT39', 'ATT40', 'ATT18', 'ATT94',
                               'ATT125', 'ATT128']
    m.wrf_feature_list_value = ['ATT57', 'ATT39', 'ATT40', 'ATT18', 'ATT94', 'ATT125', 'ATT128']

print(station)
print(paramID)

print(m.test_feature_list)
print(m.feature_list_full)
print(m.wrf_feature_list_full)
#######################################################


global_path = "cache/"+m.station+"_ablation"+m.test_feature_list[0]+"_all"

_, train_y = m.read_kentucky_file(year=year, year_during=year_during, start_date=start_date, during=during, shift=5,
                                    seq=seq)
train_x = m.read_wrf_file(year=year, year_during=year_during, start_date=start_date, during=during, shift=5,
                               seq=seq)

train_x = m.sanitize(np.array(train_x))
train_y = m.sanitize(np.array(train_y))

x_scaler = MinMaxScaler()
x_scaler.fit(train_x)
train_x = x_scaler.transform(train_x)
x_train = train_x

y_scaler = MinMaxScaler()
y_scaler.fit(train_y)
train_y = y_scaler.transform(train_y)
y_train = train_y

y_train_L = y_train.reshape(1, y_train.shape[0])[0]
x_train = np.array(x_train)
y_train_L = np.array(y_train_L)

encoder_input_data, decoder_input_data, decoder_target_data = m.pre_seq(x_train, y_train_L)

_, test_y = m.read_kentucky_file(year=T_year, year_during=T_year_during, start_date=T_start_date, during=T_during,
                                  shift=5, seq=seq)
test_x = m.read_wrf_file(year=T_year, year_during=T_year_during, start_date=T_start_date, during=T_during, shift=5,
                             seq=seq)

#test_x.to_csv('test_x.csv')

test_x = m.sanitize(np.array(test_x))
test_y = m.sanitize(np.array(test_y))

test_x = x_scaler.transform(test_x)
x_test = test_x

test_y = y_scaler.transform(test_y)
y_test = test_y

y_test_L = y_test.reshape(1, y_test.shape[0])[0]
x_test = np.array(x_test)
y_test_L = np.array(y_test_L)
# print("Printing labeled data")
# print(y_test_L)
test_encoder_input_data, test_decoder_input_data, test_decoder_target_data = m.pre_seq(x_test, y_test_L)

batch_size = 64  # Batch size for training.
epochs = model_epochs  # Number of epochs to train for.
latent_dim = m.latent_dim  # Latent dimensionality of the encoding space.
# Path to the data txt file on disk.

# Vectorized the data.

input_texts = encoder_input_data
input_len = len(input_texts)
target_texts = []

#num_encoder_tokens = len(m.feature_list_full_value)

num_encoder_tokens = len(m.wrf_feature_list_value)
num_decoder_tokens = len(m.test_feature_list) + 2

max_encoder_seq_length = 12
max_decoder_seq_length = decoderLen + 1

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='sigmoid')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

if train:
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs,
              validation_split=0.2, verbose=0)
    # Save model
    model.save(global_path + '.model')

model = load_model(global_path + '.model')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states
# Define sampling models

# '''
encoder_model = Model(encoder_inputs, encoder_states)
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
print(decoder_model.summary())
if train:
    encoder_model.save(global_path + '.encoder')
    decoder_model.save(global_path + '.decoder')

encoder_model = load_model(global_path + '.encoder')
decoder_model = load_model(global_path + '.decoder')
# '''

'''
# Define sampling models
# Restore the model and construct the encoder and decoder.


encoder_inputs = model.input[0]  # input_1
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]  # input_2
decoder_state_input_h = Input(shape=(latent_dim,), name="input_3")
decoder_state_input_c = Input(shape=(latent_dim,), name="input_4")
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
#'''


def decode_sequence(input_seq, Plabel):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))

    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, 0] = Plabel

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        pre_tmp_value = output_tokens[0, 0, 0]

        decoded_sentence.append(pre_tmp_value)
        end_bit = output_tokens[0, 0, -1]

        # Exit condition: either hit max length
        # or find stop character.

        if (end_bit > 1 or len(decoded_sentence) > max_decoder_seq_length - 1):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, 0] = pre_tmp_value
        # Update states

        states_value = [h, c]

    return decoded_sentence


label_list = []
pre_list = []

for seq_index in range(1,len(test_encoder_input_data)):
    # for seq_index in range(len(test_encoder_input_data)):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = test_encoder_input_data[seq_index: seq_index+1]

    label_data = [i[0] for i in test_decoder_target_data[seq_index]]

    # added by Puru to adjust known value
    test_label = test_decoder_input_data[seq_index,0,0]

    decoded_sentence = decode_sequence(input_seq, test_label)

    if seq_index < 10:
        print('Label sentence:', label_data)
        # print('Label sentence:', decoder_target_data[seq_index])
        print('Decoded sentence:', decoded_sentence)

    label_list.append(label_data)
    pre_list.append(decoded_sentence)

print("########################")
# ans = m.draw_sequence(label_list, pre_list)
# print(ans)
# m.draw_lstm_without_e(label_list, pre_list, y_scaler, lstm_path = global_path)

# m.compute_mse_scalar(label_list, pre_list, y_scaler)
# m.compute_mse_winddir(label_list, pre_list, y_scaler)

m.draw_lstm_without_e(label_list, pre_list, y_scaler, lstm_path=global_path)

if paramID == 3:
    m.print_MSE_winddir(label_list, pre_list, y_scaler, lstm_path=global_path, decoderLen = decoderLen)
else:
    m.print_MSE(label_list, pre_list, y_scaler, lstm_path=global_path, decoderLen = decoderLen)





