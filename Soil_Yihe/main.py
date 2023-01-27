import numpy as np
from absl import app
from absl import flags

import pickle
from datetime import datetime

from weather import Weather
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

from keras.models import Model
from keras.layers import Input, LSTM, Dense, Add
from keras.models import load_model

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator
from matplotlib import rcParams
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 44
plt.rc('xtick', labelsize=44) 
plt.rc('ytick', labelsize=44)

FLAGS = flags.FLAGS
flags.DEFINE_string('task', default='1', help='the tast id')
flags.DEFINE_bool('train', default=False, help='train the model or not')


def task1():
    m = Weather()
    # soil_data = m.read_soil_data('2022-01-01', '2022-12-01')
    soil_data = m.read_soil_data('2021-01-01', '2021-12-01')
    # soil_data = m.read_soil_data('2020-01-01', '2020-12-01')

    start_date_str = soil_data.index[0].strftime("%Y-%m-%d")
    end_date_str = soil_data.index[-1].strftime("%Y-%m-%d")

    station_data = m.read_station_data(start_date_str, end_date_str)

    sensor_15 = soil_data['Sensor Reading @ 15 cm']
    sensor_30 = soil_data['Sensor Reading @ 30 cm']
    sensor_60 = soil_data['Sensor Reading @ 60 cm']
    sensor_15_g = sensor_15.groupby(sensor_15.index.strftime("%Y-%m-%d")).mean().values
    sensor_30_g = sensor_30.groupby(sensor_30.index.strftime("%Y-%m-%d")).mean().values
    sensor_60_g = sensor_60.groupby(sensor_60.index.strftime("%Y-%m-%d")).mean().values


    particitation = station_data['Precipitation (Inches n.nn)  total rain fall that occurred for the day.'].values

    df_x = sensor_15.groupby(sensor_15.index.strftime("%Y-%m-%d")).mean()
    df_x.index = pd.to_datetime(df_x.index)

    sensor = [sensor_15_g, sensor_30_g, sensor_60_g]
    label = ['15 cm', '30 cm', '60 cm']

    fig, ax1s = plt.subplots(3,1, figsize=(12,20))
    fig.suptitle('2021')
    idx = 0
    for ax1 in ax1s:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))

        lns1 = ax1.plot(df_x.index, sensor[idx], '--', linewidth=2, label=label[idx])
        plt.gcf().autofmt_xdate()
        ax2 = ax1.twinx()
        lns2 = ax2.plot(df_x.index, particitation, 'r', label='Precipitation')

        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc=0, frameon=False)
        idx += 1

    plt.savefig('2021.pdf')
    plt.show()

def task2():
    m = Weather()
    m.test_feature_list = ["Sensor Reading @ 15 cm"]
    file_label = "_t15_"
    global_path = "Cache/test"

    train_y = m.read_soil_data('2020-01-01', '2020-12-01')
    start_date_str_y = train_y.index[0].strftime("%Y-%m-%d %H:%M")
    end_date_str_y = train_y.index[-1].strftime("%Y-%m-%d %H:%M")

    train_x = m.read_station_data(start_date_str_y, end_date_str_y)
    start_date_str_x = train_x.index[0].strftime("%Y-%m-%d %H:%M")
    end_date_str_x = train_x.index[-1].strftime("%Y-%m-%d %H:%M")

    start_date = max(datetime.strptime(start_date_str_x, "%Y-%m-%d %H:%M"), datetime.strptime(start_date_str_y, "%Y-%m-%d %H:%M"))
    end_date = min(datetime.strptime(end_date_str_x, "%Y-%m-%d %H:%M"), datetime.strptime(end_date_str_y, "%Y-%m-%d %H:%M"))
    
    start_date_str = start_date.strftime("%Y-%m-%d %H:%M")
    end_date_str = end_date.strftime("%Y-%m-%d %H:%M")

    train_x = m.read_station_data(start_date_str, end_date_str)
    train_y = m.read_soil_data(start_date_str, end_date_str)

    train_x = train_x.join(train_y["Sensor Reading @ 15 cm"])

    # one row mismatch
    train_x = train_x.iloc[:-1]
    train_y = train_y.iloc[1:]

    x_scaler = MinMaxScaler()
    x_scaler.fit(train_x)
    train_x = x_scaler.transform(train_x)

    y_scaler = MinMaxScaler()
    y_scaler.fit(train_y)
    train_y = y_scaler.transform(train_y)

    # print(train_x)
    # print(train_y)


    # print(np.array(train_x[:,-2]>0.2))

    y_train_L = train_y.reshape(1, train_y.shape[0])[0]
    x_train = np.array(train_x)
    y_train_L = np.array(y_train_L)

    encoder_input_data, decoder_input_data, decoder_target_data = m.pre_seq(x_train, y_train_L, encoder_seq = 6,  decoder_seq = 6)
    
    encoder_input_data_no, decoder_input_data_no, decoder_target_data_no = [], [], []
    
    for i, j, k in zip(encoder_input_data, decoder_input_data, decoder_target_data):
        if np.array(i[:,-2]>0.2).sum()>0: # rain
            continue
        encoder_input_data_no.append(i)
        decoder_input_data_no.append(j)
        decoder_target_data_no.append(k)
    

    encoder_input_data = np.array(encoder_input_data_no)
    decoder_input_data = np.array(decoder_input_data_no)
    decoder_target_data = np.array(decoder_target_data_no)
  

    batch_size = 10  # Batch size for training.
    epochs = 50  # Number of epochs to train for.
    latent_dim = 16  # Latent dimensionality of the encoding space.
    # Path to the data txt file on disk.

    # Vectorize the data.
    input_texts = encoder_input_data
    input_len = len(input_texts)
    target_texts = []

    # return

    num_encoder_tokens = len(m.feature_list_full_value)   # remove index
    num_decoder_tokens = len(m.test_feature_list)+2
    max_encoder_seq_length = 12
    # max_decoder_seq_length = 6+1
    max_decoder_seq_length = len(decoder_target_data[0])

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


    if FLAGS.train:
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data,batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)
        # Save model
        model.save(global_path+file_label+'.model')

    model = load_model(global_path+file_label+'.model')

    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    if FLAGS.train:
        encoder_model.save(global_path+file_label+'.encoder')
        decoder_model.save(global_path+file_label+'.decoder')

    encoder_model = load_model(global_path+file_label+'.encoder')
    decoder_model = load_model(global_path+file_label+'.decoder')


    test_y = m.read_soil_data('2021-01-01', '2021-12-01')
    start_date_str = test_y.index[0].strftime("%Y-%m-%d")
    end_date_str = test_y.index[-1].strftime("%Y-%m-%d")


    test_x = m.read_station_data(start_date_str, end_date_str)


    test_x = test_x.iloc[:-5]
    test_y = test_y.iloc[3:]

    y_scaler = MinMaxScaler()
    y_scaler.fit(test_y)
    test_y = y_scaler.transform(test_y)
    y_test_L = test_y.reshape(1, test_y.shape[0])[0]
    
    

    x_scaler = MinMaxScaler()
    x_scaler.fit(test_x)
    test_x = x_scaler.transform(test_x)

    
    x_test = np.array(train_x)
    y_test_L = np.array(y_test_L)
    test_encoder_input_data, test_decoder_input_data, test_decoder_target_data =  m.pre_seq(x_test, y_test_L, encoder_seq = 6,  decoder_seq = 6)


    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))

        # Populate the first character of target sequence with the start character.
        # target_seq[0, 0, ] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
                output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

    
                pre_tmp_value = output_tokens[0,0,0]

                decoded_sentence.append(pre_tmp_value)
                end_bit = output_tokens[0,0,-1]
        
                # Exit condition: either hit max length
                # or find stop character.

                if (end_bit > 1 or len(decoded_sentence) > max_decoder_seq_length-1):
                    stop_condition = True

                # Update the target sequence (of length 1).
                target_seq = np.zeros((1, 1, num_decoder_tokens))
                target_seq[0, 0, 0] = pre_tmp_value
                # Update states

                states_value = [h, c]

        return decoded_sentence



    label_list = []
    pre_list = []

    for seq_index in range(len(test_encoder_input_data)):
    #for seq_index in range(len(test_encoder_input_data)):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        print(seq_index)
        input_seq = test_encoder_input_data[seq_index: seq_index + 1]
    
        decoded_sentence = decode_sequence(input_seq)
        label_data = [i[0] for i in  test_decoder_target_data[seq_index]]
        
        if seq_index < 10:
            print('Label sentence:', label_data)
            # print('Label sentence:', decoder_target_data[seq_index])
            print('Decoded sentence:', decoded_sentence)

        label_list.append(label_data)
        pre_list.append(decoded_sentence)

    print("########################")
    m.draw_lstm_without_e(label_list, pre_list, y_scaler, file_label, lstm_path = global_path)


def task3():

    file_label = "_t15_"
    global_path = "Cache/test"

    m = Weather()
    m.test_feature_list = ["Sensor Reading @ 15 cm"]
    soil_data = m.read_soil_data('2021-01-01', '2021-12-01')
    soil_data = soil_data.iloc[3:]
    soil_data.index = pd.to_datetime(soil_data.index)

    
    tru = pickle.load(open(global_path+file_label+'.true', "rb"))
    pre = pickle.load(open(global_path+file_label+'.pre', "rb"))

    # print(len(tru))
    # print(len(pre))
    tru_10 = []
    pre_10 = []
    for i in range(len(tru)):
        tru_10.append(tru[i][5])
        pre_10.append(pre[i][5])


    fig, ax1s = plt.subplots(4, 1, figsize=(12,30))
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))

    idx = 0
    for ax1 in ax1s:
        tru_10 = []
        pre_10 = []
        for i in range(len(tru)):
            tru_10.append(tru[i][idx])
            pre_10.append(pre[i][idx])

        lns1 = plt.plot(tru_10, 'r', label='true', linewidth=2)
        ax2 = ax1.twinx()
        lns2 = ax2.plot(pre_10, label='prediction')

        # plt.gcf().autofmt_xdate()

        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc=0, frameon=False)
        ax1.set_title(str((idx*2+2))+'h')
        ax1.get_yaxis().set_visible(False)
        idx+=1



    plt.ylabel("15 cm mositor")
    plt.xlabel("minutes")        
    plt.savefig(file_label+'.pdf')
    plt.show() 


def task4():
    m = Weather()
    m.test_feature_list = ["Sensor Reading @ 15 cm"]
    file_label = "_t15sameday_"
    global_path = "Cache/test"

    train_y = m.read_soil_data('2020-01-01', '2020-12-01')
    start_date_str_y = train_y.index[0].strftime("%Y-%m-%d %H:%M")
    end_date_str_y = train_y.index[-1].strftime("%Y-%m-%d %H:%M")

    train_x = m.read_station_data(start_date_str_y, end_date_str_y)
    start_date_str_x = train_x.index[0].strftime("%Y-%m-%d %H:%M")
    end_date_str_x = train_x.index[-1].strftime("%Y-%m-%d %H:%M")

    start_date = max(datetime.strptime(start_date_str_x, "%Y-%m-%d %H:%M"), datetime.strptime(start_date_str_y, "%Y-%m-%d %H:%M"))
    end_date = min(datetime.strptime(end_date_str_x, "%Y-%m-%d %H:%M"), datetime.strptime(end_date_str_y, "%Y-%m-%d %H:%M"))
    
    start_date_str = start_date.strftime("%Y-%m-%d %H:%M")
    end_date_str = end_date.strftime("%Y-%m-%d %H:%M")

    train_x = m.read_station_data(start_date_str, end_date_str)
    train_y = m.read_soil_data(start_date_str, end_date_str)

    train_x = train_x.join(train_y["Sensor Reading @ 15 cm"])

    # one row mismatch
    train_x = train_x.iloc[:-1]
    train_y = train_y.iloc[1:]

    x_scaler = MinMaxScaler()
    x_scaler.fit(train_x)
    train_x = x_scaler.transform(train_x)

    y_scaler = MinMaxScaler()
    y_scaler.fit(train_y)
    train_y = y_scaler.transform(train_y)


    print(np.array(train_x[:,-2]>0.2))

    y_train_L = train_y.reshape(1, train_y.shape[0])[0]
    x_train = np.array(train_x)
    y_train_L = np.array(y_train_L)

    encoder_input_data, decoder_input_data, decoder_target_data = m.pre_seq(x_train, y_train_L, encoder_seq = 6,  decoder_seq = 12)
    
    encoder_input_data_no, decoder_input_data_no, decoder_target_data_no = [], [], []
    
    for i, j, k in zip(encoder_input_data, decoder_input_data, decoder_target_data):
        if np.array(i[:,-2]>0.2).sum()>0: # rain
            continue
        encoder_input_data_no.append(i)
        decoder_input_data_no.append(j)
        decoder_target_data_no.append(k)
    

    encoder_input_data = np.array(encoder_input_data_no)
    decoder_input_data = np.array(decoder_input_data_no)
    decoder_target_data = np.array(decoder_target_data_no)

    test_encoder_input_data = encoder_input_data[800:, : ,:]
    test_decoder_input_data = decoder_input_data[800:, : ,:]
    test_decoder_target_data = decoder_target_data[800:, : ,:]

    # encoder_input_data = encoder_input_data[:800, : ,:]
    # decoder_input_data = decoder_input_data[:800, : ,:]
    # decoder_target_data = decoder_target_data[:800, : ,:]



    batch_size = 10  # Batch size for training.
    epochs = 100  # Number of epochs to train for.
    latent_dim = 16  # Latent dimensionality of the encoding space.
    # Path to the data txt file on disk.

    # Vectorize the data.
    input_texts = encoder_input_data
    input_len = len(input_texts)
    target_texts = []



    num_encoder_tokens = len(m.feature_list_full_value)   # remove index
    num_decoder_tokens = len(m.test_feature_list)+2
    max_encoder_seq_length = 12
    # max_decoder_seq_length = 6+1
    max_decoder_seq_length = len(decoder_target_data[0])

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


    if FLAGS.train:
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data,batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)
        # Save model
        model.save(global_path+file_label+'.model')

    model = load_model(global_path+file_label+'.model')

    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    if FLAGS.train:
        encoder_model.save(global_path+file_label+'.encoder')
        decoder_model.save(global_path+file_label+'.decoder')

    encoder_model = load_model(global_path+file_label+'.encoder')
    decoder_model = load_model(global_path+file_label+'.decoder')


    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))

        # Populate the first character of target sequence with the start character.
        # target_seq[0, 0, ] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
                output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

    
                pre_tmp_value = output_tokens[0,0,0]

                decoded_sentence.append(pre_tmp_value)
                end_bit = output_tokens[0,0,-1]
        
                # Exit condition: either hit max length
                # or find stop character.

                if (end_bit > 1 or len(decoded_sentence) > max_decoder_seq_length-1):
                    stop_condition = True

                # Update the target sequence (of length 1).
                target_seq = np.zeros((1, 1, num_decoder_tokens))
                target_seq[0, 0, 0] = pre_tmp_value
                # Update states

                states_value = [h, c]

        return decoded_sentence



    label_list = []
    pre_list = []

    for seq_index in range(len(test_encoder_input_data)):
    #for seq_index in range(len(test_encoder_input_data)):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        print(seq_index)
        input_seq = test_encoder_input_data[seq_index: seq_index + 1]
    
        decoded_sentence = decode_sequence(input_seq)
        label_data = [i[0] for i in  test_decoder_target_data[seq_index]]
        
        if seq_index < 10:
            print('Label sentence:', label_data)
            # print('Label sentence:', decoder_target_data[seq_index])
            print('Decoded sentence:', decoded_sentence)

        label_list.append(label_data)
        pre_list.append(decoded_sentence)

    print("########################")
    m.draw_lstm_without_e(label_list, pre_list, y_scaler, file_label, lstm_path = global_path)


def task5():

    file_label = "_t15sameday_"
    global_path = "Cache/test"

    # m = Weather()
    # m.test_feature_list = ["Sensor Reading @ 15 cm"]
    # soil_data = m.read_soil_data('2021-01-01', '2021-12-01')
    # soil_data = soil_data.iloc[3:]
    # soil_data.index = pd.to_datetime(soil_data.index)

    
    tru = pickle.load(open(global_path+file_label+'.true', "rb"))
    pre = pickle.load(open(global_path+file_label+'.pre', "rb"))

    # print(len(tru))
    # print(len(pre))
    # tru_10 = []
    # pre_10 = []
    # for i in range(len(tru)):
    #     tru_10.append(tru[i][5])
    #     pre_10.append(pre[i][5])


    fig, ax1s = plt.subplots(4, 1, figsize=(12,30))
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))

    idx = 0
    for ax1 in ax1s:
        tru_10 = []
        pre_10 = []
        for i in range(len(tru)):
            tru_10.append(tru[i][idx])
            pre_10.append(pre[i][idx])

        lns1 = plt.plot(tru_10, 'r', label='true', linewidth=2)
        ax2 = ax1.twinx()
        lns2 = ax2.plot(pre_10, label='prediction')

        # plt.gcf().autofmt_xdate()

        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc=0, frameon=False)
        ax1.set_title(str((idx*2+2))+'h')
        ax1.get_yaxis().set_visible(False)
        idx+=1



    plt.ylabel("15 cm mositor")
    plt.xlabel("minutes")        
    plt.savefig(file_label+'.pdf')
    plt.show() 


def task6():

    file_label = "_t15sameday_"
    global_path = "Cache/test"


    
    tru = pickle.load(open(global_path+file_label+'.true', "rb"))
    pre = pickle.load(open(global_path+file_label+'.pre', "rb"))

    # start_point = [3, 150 ,160, 180]
    start_point = [20, 60 ,80, 100]



    fig, ax1s = plt.subplots(4, 1, figsize=(12,30))

    idx = 0
    for ax1 in ax1s:
        past = []
        tru_10 = []
        pre_10 = []
        
        x_past = []
        x_tre = []
        x_pre = []
        for i in range(start_point[idx]+1):
            past.append(tru[i][0])
            x_past.append(i)
        x_tre = [i for i in range(x_past[-1], x_past[-1]+49)]
        tru_10 = tru[start_point[idx]]
        pre_10 = pre[start_point[idx]]

        print(pre)
        # print(pre[0])
        ax1.plot(x_past, past,'--', c='r', linewidth=2)
        lns1 = ax1.plot(x_tre, tru_10, c='r', linewidth=2, label='True')
        
    #     lns1 = plt.plot(tru_10, 'r', label='true', linewidth=2)
        # ax2 = ax1.twinx()
        lns2 = ax1.plot(x_tre, pre_10, label='Prediction')
    #     lns2 = ax2.plot(pre_10, label='prediction')

    #     # plt.gcf().autofmt_xdate()

        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0, frameon=False)
    #     ax1.set_title(str((idx*2+2))+'h')
    #     ax1.get_yaxis().set_visible(False)
        idx+=1



    # plt.ylabel("15 cm mositor")
    # plt.xlabel("minutes")        
    plt.savefig(file_label+'_time.pdf')
    plt.show()


def task7():
    m = Weather()
    m.test_feature_list = ["Sensor Reading @ 15 cm"]
    file_label = "_t15sameday_"
    global_path = "Cache/test"

    train_year = ['2020', '2021']

    encoder_input_data_all = []
    decoder_input_data_all = []
    decoder_target_data_all = []

    for y in train_year:
        train_y = m.read_soil_data(y+'-01-01', y+'-12-01')
        start_date_str_y = train_y.index[0].strftime("%Y-%m-%d %H:%M")
        end_date_str_y = train_y.index[-1].strftime("%Y-%m-%d %H:%M")

        train_x = m.read_station_data(start_date_str_y, end_date_str_y)
        start_date_str_x = train_x.index[0].strftime("%Y-%m-%d %H:%M")
        end_date_str_x = train_x.index[-1].strftime("%Y-%m-%d %H:%M")

        start_date = max(datetime.strptime(start_date_str_x, "%Y-%m-%d %H:%M"), datetime.strptime(start_date_str_y, "%Y-%m-%d %H:%M"))
        end_date = min(datetime.strptime(end_date_str_x, "%Y-%m-%d %H:%M"), datetime.strptime(end_date_str_y, "%Y-%m-%d %H:%M"))
        
        start_date_str = start_date.strftime("%Y-%m-%d %H:%M")
        end_date_str = end_date.strftime("%Y-%m-%d %H:%M")

        train_x = m.read_station_data(start_date_str, end_date_str)
        train_y = m.read_soil_data(start_date_str, end_date_str)

        train_x = train_x.join(train_y["Sensor Reading @ 15 cm"])


        # train_x = train_x.iloc[:-1]
        # train_y = train_y.iloc[1:]

        x_scaler = MinMaxScaler()
        x_scaler.fit(train_x)
        train_x = x_scaler.transform(train_x)

        y_scaler = MinMaxScaler()
        y_scaler.fit(train_y)
        train_y = y_scaler.transform(train_y)



        y_train_L = train_y.reshape(1, train_y.shape[0])[0]
        x_train = np.array(train_x)
        y_train_L = np.array(y_train_L)

        encoder_input_data, decoder_input_data, decoder_target_data = m.pre_seq(x_train, y_train_L, encoder_seq = 48,  decoder_seq = 48)
        
        # encoder_input_data_no, decoder_input_data_no, decoder_target_data_no = [], [], []
        
        iid = 0
        for i, j, k in zip(encoder_input_data, decoder_input_data, decoder_target_data):
            if np.array(i[:,-2]>0.2).sum()>0: # rain
                continue
            if np.isnan(i).any() | np.isnan(j).any() | np.isnan(k).any():
                continue
            # print(i)
            # print(j)
            # print(k)
            # print("==================================")
            # if iid == 2:
            #     return
            encoder_input_data_all.append(i)
            decoder_input_data_all.append(j)
            decoder_target_data_all.append(k)
            iid += 1

        # encoder_input_data_all = np.vstack([encoder_input_data_all, encoder_input_data])
        # decoder_input_data_all = np.vstack([encoder_input_data_all, decoder_input_data])
        # decoder_target_data_all = np.vstack([decoder_target_data_all, decoder_target_data])

    encoder_input_data = np.array(encoder_input_data_all)
    decoder_input_data = np.array(decoder_input_data_all)
    decoder_target_data = np.array(decoder_target_data_all)

    # print(encoder_input_data)
    # print(decoder_input_data)
    # print(decoder_target_data)
    # return

    # print(encoder_input_data)
    # print(decoder_target_data)

    # return

    batch_size = 20  # Batch size for training.
    epochs = 100  # Number of epochs to train for.
    latent_dim = 36  # Latent dimensionality of the encoding space.
    # Path to the data txt file on disk.

    # Vectorize the data.
    input_texts = encoder_input_data
    input_len = len(input_texts)
    target_texts = []



    num_encoder_tokens = len(m.feature_list_full_value)   # remove index
    num_decoder_tokens = len(m.test_feature_list)+2
    max_encoder_seq_length = 12
    # max_decoder_seq_length = 6+1
    max_decoder_seq_length = len(decoder_target_data[0])

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


    if FLAGS.train:
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data,batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)
        # Save model
        model.save(global_path+file_label+'.model')

    model = load_model(global_path+file_label+'.model')

    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    if FLAGS.train:
        encoder_model.save(global_path+file_label+'.encoder')
        decoder_model.save(global_path+file_label+'.decoder')

    encoder_model = load_model(global_path+file_label+'.encoder')
    decoder_model = load_model(global_path+file_label+'.decoder')


    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))

        # Populate the first character of target sequence with the start character.
        # target_seq[0, 0, ] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
                output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

    
                pre_tmp_value = output_tokens[0,0,0]

                decoded_sentence.append(pre_tmp_value)
                end_bit = output_tokens[0,0,-1]
        
                # Exit condition: either hit max length
                # or find stop character.

                if (end_bit > 1 or len(decoded_sentence) > max_decoder_seq_length-1):
                    stop_condition = True

                # Update the target sequence (of length 1).
                target_seq = np.zeros((1, 1, num_decoder_tokens))
                target_seq[0, 0, 0] = pre_tmp_value
                # Update states

                states_value = [h, c]

        return decoded_sentence

    test_y = m.read_soil_data('2022-01-01', '2022-12-01')
    start_date_str = test_y.index[0].strftime("%Y-%m-%d")
    end_date_str = test_y.index[-1].strftime("%Y-%m-%d")

    test_x = m.read_station_data(start_date_str, end_date_str)

    test_x = test_x.join(test_y["Sensor Reading @ 15 cm"])

    test_x = test_x.iloc[:-1]
    test_y = test_y.iloc[1:]

    y_scaler = MinMaxScaler()
    y_scaler.fit(test_y)
    test_y = y_scaler.transform(test_y)
    y_test_L = test_y.reshape(1, test_y.shape[0])[0]
    
    

    x_scaler = MinMaxScaler()
    x_scaler.fit(test_x)
    test_x = x_scaler.transform(test_x)

    
    x_test = np.array(test_x)
    y_test_L = np.array(y_test_L)
    test_encoder_input_data, test_decoder_input_data, test_decoder_target_data =  m.pre_seq(x_test, y_test_L, encoder_seq = 48,  decoder_seq = 48)


    label_list = []
    pre_list = []

    for seq_index in range(len(test_encoder_input_data)):
    #for seq_index in range(len(test_encoder_input_data)):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        print(seq_index)
        input_seq = test_encoder_input_data[seq_index: seq_index + 1]
    
        decoded_sentence = decode_sequence(input_seq)
        label_data = [i[0] for i in  test_decoder_target_data[seq_index]]
        
        if seq_index < 10:
            print('Label sentence:', label_data)
            # print('Label sentence:', decoder_target_data[seq_index])
            print('Decoded sentence:', decoded_sentence)

        label_list.append(label_data)
        pre_list.append(decoded_sentence)

    print("########################")
    m.draw_lstm_without_e(label_list, pre_list, y_scaler, file_label, lstm_path = global_path)



def task8():
    m = Weather()
    m.test_feature_list = ["Sensor Reading @ 15 cm"]
    file_label = "_t15self_"
    global_path = "Cache/test"

    train_year = ['2020', '2021']

    encoder_input_data_all = []
    decoder_input_data_all = []
    decoder_target_data_all = []

    for y in train_year:
        train_y = m.read_soil_data(y+'-01-01', y+'-12-01')
        train_x = train_y

        # train_x = train_x.join(train_y["Sensor Reading @ 15 cm"])


        train_x = train_x.iloc[:-1]
        train_y = train_y.iloc[1:]

        x_scaler = MinMaxScaler()
        x_scaler.fit(train_x)
        train_x = x_scaler.transform(train_x)

        y_scaler = MinMaxScaler()
        y_scaler.fit(train_y)
        train_y = y_scaler.transform(train_y)



        y_train_L = train_y.reshape(1, train_y.shape[0])[0]
        x_train = np.array(train_x)
        y_train_L = np.array(y_train_L)

        encoder_input_data, decoder_input_data, decoder_target_data = m.pre_seq(x_train, y_train_L, encoder_seq = 12,  decoder_seq = 12)
        
        # encoder_input_data_no, decoder_input_data_no, decoder_target_data_no = [], [], []
        
        iid = 0
        for i, j, k in zip(encoder_input_data, decoder_input_data, decoder_target_data):
            # if np.array(i[:,-2]>0.2).sum()>0: # rain
            #     continue
            if np.isnan(i).any() | np.isnan(j).any() | np.isnan(k).any():
                continue
            # print(i)
            # print(j)
            # print(k)
            # print("==================================")
            # if iid == 2:
            #     return
            encoder_input_data_all.append(i)
            decoder_input_data_all.append(j)
            decoder_target_data_all.append(k)
            iid += 1

        # encoder_input_data_all = np.vstack([encoder_input_data_all, encoder_input_data])
        # decoder_input_data_all = np.vstack([encoder_input_data_all, decoder_input_data])
        # decoder_target_data_all = np.vstack([decoder_target_data_all, decoder_target_data])

    encoder_input_data = np.array(encoder_input_data_all)
    decoder_input_data = np.array(decoder_input_data_all)
    decoder_target_data = np.array(decoder_target_data_all)

    # print(encoder_input_data)
    # print(decoder_target_data)
    # return

    # print(encoder_input_data)
    # print(decoder_input_data)
    # print(decoder_target_data)
    # return

    # print(encoder_input_data)
    # print(decoder_target_data)

    # return

    batch_size = 20  # Batch size for training.
    epochs = 100  # Number of epochs to train for.
    latent_dim = 16  # Latent dimensionality of the encoding space.
    # Path to the data txt file on disk.

    # Vectorize the data.
    input_texts = encoder_input_data
    input_len = len(input_texts)
    target_texts = []



    num_encoder_tokens = 1
    num_decoder_tokens = len(m.test_feature_list)+2
    max_encoder_seq_length = 12
    # max_decoder_seq_length = 6+1
    max_decoder_seq_length = len(decoder_target_data[0])

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


    if FLAGS.train:
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data,batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)
        # Save model
        model.save(global_path+file_label+'.model')

    model = load_model(global_path+file_label+'.model')

    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    if FLAGS.train:
        encoder_model.save(global_path+file_label+'.encoder')
        decoder_model.save(global_path+file_label+'.decoder')

    encoder_model = load_model(global_path+file_label+'.encoder')
    decoder_model = load_model(global_path+file_label+'.decoder')


    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))

        # Populate the first character of target sequence with the start character.
        # target_seq[0, 0, ] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
                output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

    
                pre_tmp_value = output_tokens[0,0,0]

                decoded_sentence.append(pre_tmp_value)
                end_bit = output_tokens[0,0,-1]
        
                # Exit condition: either hit max length
                # or find stop character.

                if (end_bit > 1 or len(decoded_sentence) > max_decoder_seq_length-1):
                    stop_condition = True

                # Update the target sequence (of length 1).
                target_seq = np.zeros((1, 1, num_decoder_tokens))
                target_seq[0, 0, 0] = pre_tmp_value
                # Update states

                states_value = [h, c]

        return decoded_sentence

    test_y = m.read_soil_data('2022-01-01', '2022-12-01')
    test_x = test_y

    test_x = test_x.iloc[:-1]
    test_y = test_y.iloc[1:]

    y_scaler = MinMaxScaler()
    y_scaler.fit(test_y)
    test_y = y_scaler.transform(test_y)
    y_test_L = test_y.reshape(1, test_y.shape[0])[0]
    
    

    x_scaler = MinMaxScaler()
    x_scaler.fit(test_x)
    test_x = x_scaler.transform(test_x)

    
    x_test = np.array(test_x)
    y_test_L = np.array(y_test_L)
    test_encoder_input_data, test_decoder_input_data, test_decoder_target_data =  m.pre_seq(x_test, y_test_L, encoder_seq = 12,  decoder_seq = 12)


    label_list = []
    pre_list = []

    for seq_index in range(len(test_encoder_input_data)):
    #for seq_index in range(len(test_encoder_input_data)):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        print(seq_index)
        input_seq = test_encoder_input_data[seq_index: seq_index + 1]
    
        decoded_sentence = decode_sequence(input_seq)
        label_data = [i[0] for i in  test_decoder_target_data[seq_index]]
        
        if seq_index < 10:
            print('Label sentence:', label_data)
            # print('Label sentence:', decoder_target_data[seq_index])
            print('Decoded sentence:', decoded_sentence)

        label_list.append(label_data)
        pre_list.append(decoded_sentence)

    print("########################")
    m.draw_lstm_without_e(label_list, pre_list, y_scaler, file_label, lstm_path = global_path)


def task9():
    m = Weather()
    m.test_feature_list = ["Sensor Reading @ 15 cm"]
    file_label = "_t15sameday_"
    global_path = "Cache/test"

    train_year = ['2020', '2021']

    encoder_input_data_all = []
    decoder_input_data_all = []
    decoder_target_data_all = []

    for y in train_year:
        train_y = m.read_soil_data(y+'-01-01', y+'-12-01')
        start_date_str_y = train_y.index[0].strftime("%Y-%m-%d %H:%M")
        end_date_str_y = train_y.index[-1].strftime("%Y-%m-%d %H:%M")

        train_x = m.read_station_data(start_date_str_y, end_date_str_y)
        start_date_str_x = train_x.index[0].strftime("%Y-%m-%d %H:%M")
        end_date_str_x = train_x.index[-1].strftime("%Y-%m-%d %H:%M")

        start_date = max(datetime.strptime(start_date_str_x, "%Y-%m-%d %H:%M"), datetime.strptime(start_date_str_y, "%Y-%m-%d %H:%M"))
        end_date = min(datetime.strptime(end_date_str_x, "%Y-%m-%d %H:%M"), datetime.strptime(end_date_str_y, "%Y-%m-%d %H:%M"))
        
        start_date_str = start_date.strftime("%Y-%m-%d %H:%M")
        end_date_str = end_date.strftime("%Y-%m-%d %H:%M")

        train_x = m.read_station_data(start_date_str, end_date_str)
        train_y = m.read_soil_data(start_date_str, end_date_str)

        train_x = train_x.join(train_y["Sensor Reading @ 15 cm"])


        # train_x = train_x.iloc[:-1]
        # train_y = train_y.iloc[1:]

        x_scaler = MinMaxScaler()
        x_scaler.fit(train_x)
        train_x = x_scaler.transform(train_x)

        y_scaler = MinMaxScaler()
        y_scaler.fit(train_y)
        train_y = y_scaler.transform(train_y)



        y_train_L = train_y.reshape(1, train_y.shape[0])[0]
        x_train = np.array(train_x)
        y_train_L = np.array(y_train_L)

        encoder_input_data, decoder_input_data, decoder_target_data = m.pre_seq(x_train, y_train_L, encoder_seq = 48,  decoder_seq = 48)
        
        # encoder_input_data_no, decoder_input_data_no, decoder_target_data_no = [], [], []
        
        iid = 0
        for i, j, k in zip(encoder_input_data, decoder_input_data, decoder_target_data):
            if np.array(i[:,-2]>0.2).sum()>0: # rain
                continue
            if np.isnan(i).any() | np.isnan(j).any() | np.isnan(k).any():
                continue
            # print(i)
            # print(j)
            # print(k)
            # print("==================================")
            # if iid == 2:
            #     return
            encoder_input_data_all.append(i)
            decoder_input_data_all.append(j)
            decoder_target_data_all.append(k)
            iid += 1

        # encoder_input_data_all = np.vstack([encoder_input_data_all, encoder_input_data])
        # decoder_input_data_all = np.vstack([encoder_input_data_all, decoder_input_data])
        # decoder_target_data_all = np.vstack([decoder_target_data_all, decoder_target_data])

    encoder_input_data = np.array(encoder_input_data_all)
    decoder_input_data = np.array(decoder_input_data_all)
    decoder_target_data = np.array(decoder_target_data_all)

    # print(encoder_input_data)
    # print(decoder_input_data)
    # print(decoder_target_data)
    # return

    # print(encoder_input_data)
    # print(decoder_target_data)

    # return

    batch_size = 20  # Batch size for training.
    epochs = 100  # Number of epochs to train for.
    latent_dim = 36  # Latent dimensionality of the encoding space.
    # Path to the data txt file on disk.

    # Vectorize the data.
    input_texts = encoder_input_data
    input_len = len(input_texts)
    target_texts = []



    num_encoder_tokens = len(m.feature_list_full_value)   # remove index
    num_decoder_tokens = len(m.test_feature_list)+2
    max_encoder_seq_length = 12
    # max_decoder_seq_length = 6+1
    max_decoder_seq_length = len(decoder_target_data[0])

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


    if FLAGS.train:
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data,batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)
        # Save model
        model.save(global_path+file_label+'.model')

    model = load_model(global_path+file_label+'.model')

    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    if FLAGS.train:
        encoder_model.save(global_path+file_label+'.encoder')
        decoder_model.save(global_path+file_label+'.decoder')

    encoder_model = load_model(global_path+file_label+'.encoder')
    decoder_model = load_model(global_path+file_label+'.decoder')


    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))

        # Populate the first character of target sequence with the start character.
        # target_seq[0, 0, ] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = []
        while not stop_condition:
                output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

    
                pre_tmp_value = output_tokens[0,0,0]

                decoded_sentence.append(pre_tmp_value)
                end_bit = output_tokens[0,0,-1]
        
                # Exit condition: either hit max length
                # or find stop character.

                if (end_bit > 1 or len(decoded_sentence) > max_decoder_seq_length-1):
                    stop_condition = True

                # Update the target sequence (of length 1).
                target_seq = np.zeros((1, 1, num_decoder_tokens))
                target_seq[0, 0, 0] = pre_tmp_value
                # Update states

                states_value = [h, c]

        return decoded_sentence

    test_y = m.read_soil_data('2022-01-01', '2022-12-01')
    start_date_str = test_y.index[0].strftime("%Y-%m-%d")
    end_date_str = test_y.index[-1].strftime("%Y-%m-%d")

    test_x = m.read_station_data(start_date_str, end_date_str)

    test_x = test_x.join(test_y["Sensor Reading @ 15 cm"])

    test_x = test_x.iloc[:-1]
    test_y = test_y.iloc[1:]

    y_scaler = MinMaxScaler()
    y_scaler.fit(test_y)
    test_y = y_scaler.transform(test_y)
    y_test_L = test_y.reshape(1, test_y.shape[0])[0]
    
    

    x_scaler = MinMaxScaler()
    x_scaler.fit(test_x)
    test_x = x_scaler.transform(test_x)

    
    x_test = np.array(test_x)
    y_test_L = np.array(y_test_L)
    test_encoder_input_data, test_decoder_input_data, test_decoder_target_data =  m.pre_seq(x_test, y_test_L, encoder_seq = 48,  decoder_seq = 48)


    label_list = []
    pre_list = []

    for seq_index in range(len(test_encoder_input_data)):
    #for seq_index in range(len(test_encoder_input_data)):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        print(seq_index)
        input_seq = test_encoder_input_data[seq_index: seq_index + 1]
    
        decoded_sentence = decode_sequence(input_seq)
        label_data = [i[0] for i in  test_decoder_target_data[seq_index]]
        
        if seq_index < 10:
            print('Label sentence:', label_data)
            # print('Label sentence:', decoder_target_data[seq_index])
            print('Decoded sentence:', decoded_sentence)

        label_list.append(label_data)
        pre_list.append(decoded_sentence)

    print("########################")
    m.draw_lstm_without_e(label_list, pre_list, y_scaler, file_label, lstm_path = global_path)



def task10():

    for i in range(30):
        m = Weather()
        m.test_feature_list = ["Sensor Reading @ 15 cm"]
        test_name = '15'
        file_label = "_t"+test_name+"_2year/_"+str(i)
        global_path = "Cache/test"

        train_year = ['2020', '2021', '2022']


        tmp_x = []
        tmp_y = []

        for y in train_year:
            train_y = m.read_soil_data(y+'-01-01', y+'-12-01')
            start_date_str_y = train_y.index[0].strftime("%Y-%m-%d %H:%M")
            end_date_str_y = train_y.index[-1].strftime("%Y-%m-%d %H:%M")

            train_x = m.read_station_data(start_date_str_y, end_date_str_y)
            start_date_str_x = train_x.index[0].strftime("%Y-%m-%d %H:%M")
            end_date_str_x = train_x.index[-1].strftime("%Y-%m-%d %H:%M")

            start_date = max(datetime.strptime(start_date_str_x, "%Y-%m-%d %H:%M"), datetime.strptime(start_date_str_y, "%Y-%m-%d %H:%M"))
            end_date = min(datetime.strptime(end_date_str_x, "%Y-%m-%d %H:%M"), datetime.strptime(end_date_str_y, "%Y-%m-%d %H:%M"))
            
            start_date_str = start_date.strftime("%Y-%m-%d %H:%M")
            end_date_str = end_date.strftime("%Y-%m-%d %H:%M")

            train_x = m.read_station_data(start_date_str, end_date_str)
            train_y = m.read_soil_data(start_date_str, end_date_str)
            train_x = train_x.join(train_y[m.test_feature_list[0]])

            tmp_x.append(train_x)
            tmp_y.append(train_y)

        
        train_x_to = pd.concat(tmp_x, axis=0)
        train_y_to = pd.concat(tmp_y, axis=0)

        x_scaler = MinMaxScaler()
        x_scaler.fit(train_x_to)

        y_scaler = MinMaxScaler()
        y_scaler.fit(train_y_to)

        encoder_input_data_all = []
        decoder_input_data_all = []
        decoder_target_data_all = []

        for tx, ty in zip(tmp_x, tmp_y):

            train_x = tx.iloc[:-1]
            train_y = ty.iloc[1:]

            train_x = x_scaler.transform(train_x)
            train_y = y_scaler.transform(train_y)

            y_train_L = train_y.reshape(1, train_y.shape[0])[0]

            x_train = np.array(train_x)
            y_train_L = np.array(y_train_L)

            encoder_input_data, decoder_input_data, decoder_target_data = m.pre_seq(x_train, y_train_L, encoder_seq = 12,  decoder_seq = 12)
            
            iid = 0
            for i, j, k in zip(encoder_input_data, decoder_input_data, decoder_target_data):
                if np.array(i[:,-2]>0.2).sum()>0: # rain
                    continue
                if np.isnan(i).any() | np.isnan(j).any() | np.isnan(k).any():
                    continue

                encoder_input_data_all.append(i)
                decoder_input_data_all.append(j)
                decoder_target_data_all.append(k)
                iid += 1

        encoder_input_data = np.array(encoder_input_data_all)
        decoder_input_data = np.array(decoder_input_data_all)
        decoder_target_data = np.array(decoder_target_data_all)


        test_encoder_input_data = encoder_input_data[-800:-400, : ,:]
        test_decoder_input_data = decoder_input_data[-800:-400, : ,:]
        test_decoder_target_data = decoder_target_data[-800:-400, : ,:]

        encoder_input_data = encoder_input_data[:-800, : ,:]
        decoder_input_data = decoder_input_data[:-800, : ,:]
        decoder_target_data = decoder_target_data[:-800, : ,:]


        batch_size = 20  # Batch size for training.
        epochs = 100  # Number of epochs to train for.
        latent_dim = 36  # Latent dimensionality of the encoding space.
        # Path to the data txt file on disk.

        # Vectorize the data.
        input_texts = encoder_input_data
        input_len = len(input_texts)
        target_texts = []



        num_encoder_tokens = len(m.feature_list_full_value)   # remove index
        num_decoder_tokens = len(m.test_feature_list)+2
        max_encoder_seq_length = 12
        # max_decoder_seq_length = 6+1
        max_decoder_seq_length = len(decoder_target_data[0])

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


        if FLAGS.train:
            model.fit([encoder_input_data, decoder_input_data], decoder_target_data,batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)
            # Save model
            model.save(global_path+file_label+'.model')

        model = load_model(global_path+file_label+'.model')

        encoder_model = Model(encoder_inputs, encoder_states)
        decoder_state_input_h = Input(shape=(latent_dim,))
        decoder_state_input_c = Input(shape=(latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        if FLAGS.train:
            encoder_model.save(global_path+file_label+'.encoder')
            decoder_model.save(global_path+file_label+'.decoder')

        encoder_model = load_model(global_path+file_label+'.encoder')
        decoder_model = load_model(global_path+file_label+'.decoder')


        def decode_sequence(input_seq):
            # Encode the input as state vectors.
            states_value = encoder_model.predict(input_seq)

            # Generate empty target sequence of length 1.
            target_seq = np.zeros((1, 1, num_decoder_tokens))

            # Populate the first character of target sequence with the start character.
            # target_seq[0, 0, ] = 1.

            # Sampling loop for a batch of sequences
            # (to simplify, here we assume a batch of size 1).
            stop_condition = False
            decoded_sentence = []
            while not stop_condition:
                    output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        
                    pre_tmp_value = output_tokens[0,0,0]

                    decoded_sentence.append(pre_tmp_value)
                    end_bit = output_tokens[0,0,-1]
            
                    # Exit condition: either hit max length
                    # or find stop character.

                    if (end_bit > 1 or len(decoded_sentence) > max_decoder_seq_length-1):
                        stop_condition = True

                    # Update the target sequence (of length 1).
                    target_seq = np.zeros((1, 1, num_decoder_tokens))
                    target_seq[0, 0, 0] = pre_tmp_value
                    # Update states

                    states_value = [h, c]

            return decoded_sentence

        label_list = []
        pre_list = []

        for seq_index in range(len(test_encoder_input_data)):
        #for seq_index in range(len(test_encoder_input_data)):
            # Take one sequence (part of the training set)
            # for trying out decoding.
            print(seq_index)
            input_seq = test_encoder_input_data[seq_index: seq_index + 1]
        
            decoded_sentence = decode_sequence(input_seq)
            label_data = [i[0] for i in  test_decoder_target_data[seq_index]]
            
            if seq_index < 10:
                print('Label sentence:', label_data)
                # print('Label sentence:', decoder_target_data[seq_index])
                print('Decoded sentence:', decoded_sentence)

            label_list.append(label_data)
            pre_list.append(decoded_sentence)

        print("########################")
        m.draw_lstm_without_e(label_list, pre_list, y_scaler, file_label, lstm_path = global_path)

def task11():

    file_label = "_t15_2year_"
    global_path = "Cache/test"


    tru = pickle.load(open(global_path+file_label+'.true', "rb"))
    pre = pickle.load(open(global_path+file_label+'.pre', "rb"))




    fig, ax1s = plt.subplots(4, 1, figsize=(12,30))


    idx = 0
    for ax1 in ax1s:
        tru_10 = []
        pre_10 = []
        for i in range(len(tru)):
            tru_10.append(tru[i][idx])
            pre_10.append(pre[i][idx])

        lns1 = plt.plot(tru_10, 'r', label='true', linewidth=2)
        ax2 = ax1.twinx()
        lns2 = ax2.plot(pre_10, label='prediction')

        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc=0, frameon=False)
        ax1.set_title(str((idx*2+2))+'h')
        ax1.get_yaxis().set_visible(False)
        idx+=1

    plt.ylabel("15 cm mositor")
    plt.xlabel("minutes")        
    plt.savefig(file_label+'.pdf')
    plt.show() 


def task12():

    file_label = "_t15_2year_"
    global_path = "Cache/test"


    
    tru = pickle.load(open(global_path+file_label+'.true', "rb"))
    pre = pickle.load(open(global_path+file_label+'.pre', "rb"))

    # start_point = [3, 150 ,160, 180]
    start_point = [10, 67 ,200, 300]



    fig, ax1s = plt.subplots(4, 1, figsize=(12,30))

    idx = 0
    for ax1 in ax1s:
        past = []
        tru_10 = []
        pre_10 = []
        
        x_past = []
        x_tre = []
        x_pre = []
        for i in range(start_point[idx]+1):
            past.append(tru[i][0])
            x_past.append(i)
        x_tre = [i for i in range(x_past[-1], x_past[-1]+13)]
        tru_10 = tru[start_point[idx]]
        pre_10 = pre[start_point[idx]]

        # print(pre)

        ax1.plot(x_past, past,'--', c='r', linewidth=2)
        lns1 = ax1.plot(x_tre, tru_10, c='r', linewidth=2, label='True')
        

        lns2 = ax1.plot(x_tre, pre_10, label='Prediction')

        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=0, frameon=False)
    #     ax1.set_title(str((idx*2+2))+'h')
    #     ax1.get_yaxis().set_visible(False)
        idx+=1



    # plt.ylabel("15 cm mositor")
    # plt.xlabel("minutes")        
    plt.savefig(file_label+'_time.pdf')
    plt.show()


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def task13():
    '''
    draw average figure
    '''
    m = Weather()
    m.test_feature_list = ["Sensor Reading @ 60 cm"]
    test_name = '60'
    global_path = "Cache/test"
    start_point = [33, 77 ,110, 220]
    # 120  270 33:15
    idx = 0   
    fig, ax1s = plt.subplots(2, 2, figsize=(18,15))

    for row in ax1s:
        for ax1 in row:

            ys = []
            x = []
            past = []
            x_past = []
            for i in range(30):

                file_label = "_t"+test_name+"_2year/_"+str(i)

                y_tmp = []
                tru = pickle.load(open(global_path+file_label+'.true', "rb"))
                pre = pickle.load(open(global_path+file_label+'.pre', "rb"))

                ys.append(pre[start_point[idx]].tolist())

            for i in range(max(start_point[idx]-10,0), start_point[idx]+1):
                past.append(tru[i][0])
                x_past.append(i)
            x_tre = [i for i in range(x_past[-1], x_past[-1]+13)]
            tru_10 = tru[start_point[idx]]
            pre_10 = pre[start_point[idx]]

            #calculate confidence
            arr = np.array(ys)

            y_max = []
            y_mean = []
            y_min = []
            for y in arr.T:
                y_mean.append(np.mean(reject_outliers(y, m=2)))
                y_max.append(np.max(reject_outliers(y, m=2)))
                y_min.append(np.min(reject_outliers(y, m=2)))

            bias = y_mean[0]-tru_10[0]

            y_mean = [y-bias for y in y_mean]
            y_max = [y-bias for y in y_max]
            y_min = [y-bias for y in y_min]

            ax1.plot(x_past, past,'--', c='r', linewidth=2)
            lns1 = ax1.plot(x_tre, tru_10, c='r', linewidth=2, label='True')
            pre_10 = np.array(ys).mean(axis=0)

            lns2 = ax1.plot(x_tre, y_mean, label='Mean')
            # lns3 = ax1.plot(x_tre, y_min)
            # lns4 = ax1.plot(x_tre, y_max)
            lns5 = ax1.fill_between(x_tre, y_min, y_max, color='blue', alpha=0.1)

            lns = lns1+lns2
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc='upper left', frameon=False)
            idx += 1
            ax1.title.set_text('Case '+str(idx))
            ax1.set_ylabel(test_name+" (cm) Moisture")
            ax1.set_xlabel("Time point")

    plt.tight_layout()
    plt.savefig(global_path+file_label+'_'+test_name+'_time.pdf')
    plt.show()

#     # data = np.array(ys)

#     # print(data)

#     # print(data.mean(axis=0))

def task14():
    '''
    draw 15 cm and 30 cm
    '''
    # year = "2020"
    # m = Weather()
    # m.test_feature_list = ["Sensor Reading @ 15 cm"]

    # file_label = "_15_30_figure_"
    # global_path = "Cache/test"

    # data =  m.read_soil_data(year+'-01-01', year+'-12-01')
    # print(data)
    # y_15_list = data["Sensor Reading @ 15 cm"]

    # m.test_feature_list = ["Sensor Reading @ 30 cm"]
    # data =  m.read_soil_data(year+'-01-01', year+'-12-01')
    # print(data)
    # y_30_list = data["Sensor Reading @ 30 cm"]
    

    # fig, ax1s = plt.subplots(1,1, figsize=(12,10))
    # fig.suptitle('2021')
    # idx = 0
    # for ax1 in ax1s:
    #     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    #     plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))

    #     lns1 = ax1.plot(df_x.index, sensor[idx], '--', linewidth=2, label=label[idx])
    #     plt.gcf().autofmt_xdate()
    #     ax2 = ax1.twinx()
    #     lns2 = ax2.plot(df_x.index, particitation, 'r', label='Precipitation')

    #     lns = lns1+lns2
    #     labs = [l.get_label() for l in lns]
    #     ax2.legend(lns, labs, loc=0, frameon=False)
    #     idx += 1

    # plt.savefig('all.pdf')
    # plt.show()
    m = Weather()
    year = "2020"
    soil_data =  m.read_soil_data_o(year+'-01-01', year+'-12-01')




    sensor_15 = soil_data['Sensor Reading @ 15 cm']
    sensor_30 = soil_data['Sensor Reading @ 30 cm']
    sensor_60 = soil_data['Sensor Reading @ 60 cm']
    sensor_15_g = sensor_15.groupby(sensor_15.index.strftime("%Y-%m-%d")).mean().values
    sensor_30_g = sensor_30.groupby(sensor_30.index.strftime("%Y-%m-%d")).mean().values
    sensor_60_g = sensor_60.groupby(sensor_60.index.strftime("%Y-%m-%d")).mean().values



    df_x = sensor_15.groupby(sensor_15.index.strftime("%Y-%m-%d")).mean()
    df_x.index = pd.to_datetime(df_x.index)


    fig, ax1s = plt.subplots(1,1, figsize=(12,10))
    # fig.suptitle(year)
 

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))

    lns1 = ax1s.plot(df_x.index, sensor_15_g, '--', linewidth=3, label='at 15 cm')
    lns2 = ax1s.plot(df_x.index, sensor_30_g, 'r', linewidth=3, label='at 30 cm')
    plt.gcf().autofmt_xdate()
        # ax2 = ax1.twinx()
        # lns2 = ax2.plot(df_x.index, particitation, 'r', label='Precipitation')

    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax1s.legend(lns, labs, loc=(0.4,0.7), frameon=False, handletextpad=0.1,prop=dict(weight='bold'))
    ax1s.tick_params(labelsize=33)
    ax1s.set_ylabel("Moisture", weight='bold')
    plt.xticks(weight = 'bold')
    plt.yticks(weight = 'bold')

    print(year+'_15_30_.pdf')
    plt.savefig(year+'_15_30_.pdf')
    plt.show()

  
def task15():
    '''
    draw prediction in one figure
    '''
    m = Weather()
    m.test_feature_list = ["Sensor Reading @ 15 cm"]
    test_name = '15'
    global_path = "Cache/test"
    start_point = [0, 24 ,20, 30]
    # 120  270 33:15
    idx = 0   
    

    ys_1 = []
    ys_2 = []
    x = []
    past_1 = []
    x_past_1 = []
    past_2 = []
    x_past_2 = []
    for i in range(30):

        file_label = "_t"+test_name+"_2year/_"+str(i)

        y_tmp = []
        tru = pickle.load(open(global_path+file_label+'.true', "rb"))
        pre = pickle.load(open(global_path+file_label+'.pre', "rb"))

        ys_1.append(pre[start_point[0]].tolist())
        ys_2.append(pre[start_point[1]].tolist())


    for i in range(max(start_point[0]-10,0), start_point[0]+1):
        past_1.append(tru[i][0])
        x_past_1.append(i)
    for i in range(max(start_point[1]-12,0), start_point[1]+1):
        past_2.append(tru[i][0])
        x_past_2.append(i)



    x_tre_1 = [i for i in range(x_past_1[-1], x_past_1[-1]+13)]
    x_tre_2 = [i for i in range(x_past_2[-1], x_past_2[-1]+13)]
    tru_10_1 = tru[start_point[0]]
    pre_10_1 = pre[start_point[0]]
    tru_10_2 = tru[start_point[1]]
    pre_10_2 = pre[start_point[1]]


    # #calculate confidence
    arr_1 = np.array(ys_1)
    arr_2 = np.array(ys_2)

    y_max_1 = []
    y_mean_1 = []
    y_min_1 = []
    for y in arr_1.T:
        y_mean_1.append(np.mean(reject_outliers(y, m=1)))
        y_max_1.append(np.max(reject_outliers(y, m=1)))
        y_min_1.append(np.min(reject_outliers(y, m=1)))

    y_max_2 = []
    y_mean_2 = []
    y_min_2 = []
    for y in arr_2.T:
        y_mean_2.append(np.mean(reject_outliers(y, m=1)))
        y_max_2.append(np.max(reject_outliers(y, m=1)))
        y_min_2.append(np.min(reject_outliers(y, m=1)))

    bias_1 = y_mean_1[0]-tru_10_1[0]
    y_mean_1 = [y-bias_1 for y in y_mean_1]
    y_max_1 = [y-bias_1 for y in y_max_1]
    y_min_1 = [y-bias_1 for y in y_min_1]

    bias_2 = y_mean_2[0]-tru_10_2[0]
    y_mean_2 = [y-bias_2 for y in y_mean_2]
    y_max_2 = [y-bias_2 for y in y_max_2]
    y_min_2 = [y-bias_2 for y in y_min_2]


    fig, ax1 = plt.subplots(1, 1, figsize=(12,5))
    ax1.plot(x_past_1, past_1,'--', c='r', linewidth=2)
    ax1.plot(x_past_2, past_2,'--', c='r', linewidth=2)
    lns1 = ax1.plot(x_tre_1, tru_10_1, c='r', linewidth=2, label='True')
    # pre_10 = np.array(ys).mean(axis=0)

    lns2 = ax1.plot(x_tre_1, y_mean_1, color='blue', label='Mean')
    lns3 = ax1.fill_between(x_tre_1, y_min_1, y_max_1, color='blue', alpha=0.1)

    lns4 = ax1.plot(x_tre_2, tru_10_2, c='r', linewidth=2, label='True')
    lns5 = ax1.plot(x_tre_2, y_mean_2, color='blue', label='Mean')
    lns6 = ax1.fill_between(x_tre_2, y_min_2, y_max_2, color='blue', alpha=0.1)


    lns = lns1+lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left', frameon=False)
    idx += 1
    # ax1.title.set_text('Case '+str(idx))
    ax1.set_ylabel(test_name+" (cm) Moisture")
    ax1.set_xlabel("Time point")


    # plt.tight_layout()
    print(global_path+file_label+'_'+test_name+'_time.pdf')
    plt.savefig(global_path+file_label+'_'+test_name+'_time.pdf')
    plt.show()


def task16():
    '''
    draw prediction in one figure
    '''
    m = Weather()
    m.test_feature_list = ["Sensor Reading @ 15 cm"]
    test_name = '15'
    global_path = "Cache/test"
    start_point = [0, 24 ,20, 30]
    # 120  270 33:15
    idx = 0   
    

    ys_1 = []
    ys_2 = []

    # past_1 = []
    # x_past_1 = []
    # past_2 = []
    # x_past_2 = []
    for i in range(30):

        file_label = "_t"+test_name+"_2year/_"+str(i)

        y_tmp = []
        tru = pickle.load(open(global_path+file_label+'.true', "rb"))
        pre = pickle.load(open(global_path+file_label+'.pre', "rb"))

        ys_1.append(pre[start_point[0]].tolist())
        ys_2.append(pre[start_point[1]].tolist())


    # for i in range(max(start_point[0]-10,0), start_point[0]+1):
    #     past_1.append(tru[i][0])
    #     x_past_1.append(i)
    # for i in range(max(start_point[1]-12,0), start_point[1]+1):
    #     past_2.append(tru[i][0])
    #     x_past_2.append(i)



    # x_tre_1 = [i for i in range(x_past_1[-1], x_past_1[-1]+13)]
    # x_tre_2 = [i for i in range(x_past_2[-1], x_past_2[-1]+13)]
    x_tre_1 = [i for i in range(0, 13)]
    x_tre_2 = [i for i in range(0, 13)]
    tru_10_1 = tru[start_point[0]]
    pre_10_1 = pre[start_point[0]]
    tru_10_2 = tru[start_point[1]]
    pre_10_2 = pre[start_point[1]]


    # #calculate confidence
    arr_1 = np.array(ys_1)
    arr_2 = np.array(ys_2)

    y_max_1 = []
    y_mean_1 = []
    y_min_1 = []
    for y in arr_1.T:
        y_mean_1.append(np.mean(reject_outliers(y, m=1)))
        y_max_1.append(np.max(reject_outliers(y, m=1)))
        y_min_1.append(np.min(reject_outliers(y, m=1)))

    y_max_2 = []
    y_mean_2 = []
    y_min_2 = []
    for y in arr_2.T:
        y_mean_2.append(np.mean(reject_outliers(y, m=1)))
        y_max_2.append(np.max(reject_outliers(y, m=1)))
        y_min_2.append(np.min(reject_outliers(y, m=1)))

    bias_1 = y_mean_1[0]-tru_10_1[0]
    y_mean_1 = [y-bias_1 for y in y_mean_1]
    y_max_1 = [y-bias_1 for y in y_max_1]
    y_min_1 = [y-bias_1 for y in y_min_1]

    bias_2 = y_mean_2[0]-tru_10_2[0]
    y_mean_2 = [y-bias_2 for y in y_mean_2]
    y_max_2 = [y-bias_2 for y in y_max_2]
    y_min_2 = [y-bias_2 for y in y_min_2]


    fig, ax1 = plt.subplots(1, 1, figsize=(12,10))
    # ax1.plot(x_past_1, past_1,'--', c='r', linewidth=2)
    # ax1.plot(x_past_2, past_2,'--', c='r', linewidth=2)
    lns1 = ax1.plot(x_tre_1, tru_10_1, c='tab:orange', linewidth=2, label='1st day ground truth')
    # pre_10 = np.array(ys).mean(axis=0)

    lns2 = ax1.plot(x_tre_1, y_mean_1,'--', color='tab:blue', label='1st day prediction')
    # lns3 = ax1.fill_between(x_tre_1, y_min_1, y_max_1, color='blue', alpha=0.1)

    lns4 = ax1.plot(x_tre_2, tru_10_2, marker ='o', c='tab:orange', linewidth=2, label='3rd day ground truth')
    lns5 = ax1.plot(x_tre_2, y_mean_2,'--', marker ='o', color='tab:blue', label='3rd day prediction')
    # lns6 = ax1.fill_between(x_tre_2, y_min_2, y_max_2, color='blue', alpha=0.1)


    lns = lns1+lns2+lns4+lns5
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left', frameon=False)
    # ax1.set_ylim([60, 85])
    ax1.set_ylim([0, 60])

    ax1.set_ylabel(test_name+" (cm) Moisture")
    ax1.set_xlabel("Time point")


    # plt.tight_layout()
    print(global_path+file_label+'_'+test_name+'_time.pdf')
    plt.savefig(global_path+file_label+'_'+test_name+'_time.pdf')
    plt.show()



def task17():
    '''
    draw prediction 15, 30 in one figure
    '''
    test_name_1 = '15'
    test_name_2 = '30'
    global_path = "Cache/test"
    start_point = [0, 24 ,20, 30]
    # 120  270 33:15
    idx = 0   
    

    ys_1 = []
    ys_2 = []


    for i in range(30):

        file_label = "_t"+test_name_1+"_2year/_"+str(i)

        tru_1 = pickle.load(open(global_path+file_label+'.true', "rb"))
        pre_1 = pickle.load(open(global_path+file_label+'.pre', "rb"))

        ys_1.append(pre_1[start_point[0]].tolist())

    
    for i in range(30):
        file_label = "_t"+test_name_2+"_2year/_"+str(i)


        tru_2 = pickle.load(open(global_path+file_label+'.true', "rb"))
        pre_2 = pickle.load(open(global_path+file_label+'.pre', "rb"))

        ys_2.append(pre_2[start_point[0]].tolist())




    x_tre_1 = [i for i in range(0, 13)]
    x_tre_2 = [i for i in range(0, 13)]
    tru_10_1 = tru_1[start_point[0]]
    tru_10_2 = tru_2[start_point[0]]



    # #calculate confidence
    arr_1 = np.array(ys_1)
    arr_2 = np.array(ys_2)

    y_max_1 = []
    y_mean_1 = []
    y_min_1 = []
    for y in arr_1.T:
        y_mean_1.append(np.mean(reject_outliers(y, m=2)))
        y_max_1.append(np.max(reject_outliers(y, m=2)))
        y_min_1.append(np.min(reject_outliers(y, m=2)))

    y_max_2 = []
    y_mean_2 = []
    y_min_2 = []
    for y in arr_2.T:
        y_mean_2.append(np.mean(reject_outliers(y, m=2)))
        y_max_2.append(np.max(reject_outliers(y, m=2)))
        y_min_2.append(np.min(reject_outliers(y, m=2)))

    bias_1 = y_mean_1[0]-tru_10_1[0]
    y_mean_1 = [y-bias_1 for y in y_mean_1]
    y_max_1 = [y-bias_1 for y in y_max_1]
    y_min_1 = [y-bias_1 for y in y_min_1]

    bias_2 = y_mean_2[0]-tru_10_2[0]
    y_mean_2 = [y-bias_2 for y in y_mean_2]
    y_max_2 = [y-bias_2 for y in y_max_2]
    y_min_2 = [y-bias_2 for y in y_min_2]



    fig, ax1 = plt.subplots(1, 1, figsize=(12,10))
    # ax1.plot(x_past_1, past_1,'--', c='r', linewidth=2)
    # ax1.plot(x_past_2, past_2,'--', c='r', linewidth=2)
    lns1 = ax1.plot(x_tre_1, tru_10_1, c='tab:orange', linewidth=4, label='15 cm, ground truth')
    # pre_10 = np.array(ys).mean(axis=0)

    lns2 = ax1.plot(x_tre_1, y_mean_1,'--', color='tab:blue', linewidth=4, label='15 cm, prediction')
    # lns3 = ax1.fill_between(x_tre_1, y_min_1, y_max_1, color='blue', alpha=0.1)

    lns4 = ax1.plot(x_tre_2, tru_10_2, marker ='o',  markersize=10, c='tab:orange', linewidth=4, label='30 cm, ground truth')
    lns5 = ax1.plot(x_tre_2, y_mean_2,'--', marker ='o',  markersize=10, color='tab:blue', linewidth=4, label='30 cm, prediction')
    # lns6 = ax1.fill_between(x_tre_2, y_min_2, y_max_2, color='blue', alpha=0.1)


    lns = lns1+lns2+lns4+lns5
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, frameon=False, loc=(0.1, 0.35),prop=dict(weight='bold'))
    # ax1.set_ylim([60, 85])
    ax1.set_ylim([0, 80])
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    ax1.set_ylabel("Moisture", weight='bold')
    ax1.set_xlabel("Time point (2 hour apart)",fontsize=33, weight='bold')
 
    plt.xticks(weight = 'bold')
    plt.yticks(weight = 'bold')
    # ax1.set_xlabel('some xlabel', )


    # plt.tight_layout()
    print(global_path+file_label+'_'+test_name_1+'_time.pdf')
    plt.savefig(global_path+file_label+'_'+test_name_1+'_time.pdf')
    plt.show()



def task18():

    m = Weather()
    co2_data = m.read_co2(order=1)

    print(co2_data)

    tmp_data = m.read_co2(order=2)

    print(tmp_data.columns)

    # for i in range(30):
    #     m = Weather()
    #     m.test_feature_list = ["Sensor Reading @ 15 cm"]
    #     test_name = '15'
    #     file_label = "_t"+test_name+"_2year/_"+str(i)
    #     global_path = "Cache/test"

    #     train_year = ['2020', '2021', '2022']


    #     tmp_x = []
    #     tmp_y = []

    #     for y in train_year:
    #         train_y = m.read_soil_data(y+'-01-01', y+'-12-01')
    #         start_date_str_y = train_y.index[0].strftime("%Y-%m-%d %H:%M")
    #         end_date_str_y = train_y.index[-1].strftime("%Y-%m-%d %H:%M")

    #         train_x = m.read_station_data(start_date_str_y, end_date_str_y)
    #         start_date_str_x = train_x.index[0].strftime("%Y-%m-%d %H:%M")
    #         end_date_str_x = train_x.index[-1].strftime("%Y-%m-%d %H:%M")

    #         start_date = max(datetime.strptime(start_date_str_x, "%Y-%m-%d %H:%M"), datetime.strptime(start_date_str_y, "%Y-%m-%d %H:%M"))
    #         end_date = min(datetime.strptime(end_date_str_x, "%Y-%m-%d %H:%M"), datetime.strptime(end_date_str_y, "%Y-%m-%d %H:%M"))
            
    #         start_date_str = start_date.strftime("%Y-%m-%d %H:%M")
    #         end_date_str = end_date.strftime("%Y-%m-%d %H:%M")

    #         train_x = m.read_station_data(start_date_str, end_date_str)
    #         train_y = m.read_soil_data(start_date_str, end_date_str)
    #         train_x = train_x.join(train_y[m.test_feature_list[0]])

    #         tmp_x.append(train_x)
    #         tmp_y.append(train_y)

        
    #     train_x_to = pd.concat(tmp_x, axis=0)
    #     train_y_to = pd.concat(tmp_y, axis=0)

    #     x_scaler = MinMaxScaler()
    #     x_scaler.fit(train_x_to)

    #     y_scaler = MinMaxScaler()
    #     y_scaler.fit(train_y_to)

    #     encoder_input_data_all = []
    #     decoder_input_data_all = []
    #     decoder_target_data_all = []

    #     for tx, ty in zip(tmp_x, tmp_y):

    #         train_x = tx.iloc[:-1]
    #         train_y = ty.iloc[1:]

    #         train_x = x_scaler.transform(train_x)
    #         train_y = y_scaler.transform(train_y)

    #         y_train_L = train_y.reshape(1, train_y.shape[0])[0]

    #         x_train = np.array(train_x)
    #         y_train_L = np.array(y_train_L)

    #         encoder_input_data, decoder_input_data, decoder_target_data = m.pre_seq(x_train, y_train_L, encoder_seq = 12,  decoder_seq = 12)
            
    #         iid = 0
    #         for i, j, k in zip(encoder_input_data, decoder_input_data, decoder_target_data):
    #             if np.array(i[:,-2]>0.2).sum()>0: # rain
    #                 continue
    #             if np.isnan(i).any() | np.isnan(j).any() | np.isnan(k).any():
    #                 continue

    #             encoder_input_data_all.append(i)
    #             decoder_input_data_all.append(j)
    #             decoder_target_data_all.append(k)
    #             iid += 1

    #     encoder_input_data = np.array(encoder_input_data_all)
    #     decoder_input_data = np.array(decoder_input_data_all)
    #     decoder_target_data = np.array(decoder_target_data_all)


    #     test_encoder_input_data = encoder_input_data[-800:-400, : ,:]
    #     test_decoder_input_data = decoder_input_data[-800:-400, : ,:]
    #     test_decoder_target_data = decoder_target_data[-800:-400, : ,:]

    #     encoder_input_data = encoder_input_data[:-800, : ,:]
    #     decoder_input_data = decoder_input_data[:-800, : ,:]
    #     decoder_target_data = decoder_target_data[:-800, : ,:]


    #     batch_size = 20  # Batch size for training.
    #     epochs = 100  # Number of epochs to train for.
    #     latent_dim = 36  # Latent dimensionality of the encoding space.
    #     # Path to the data txt file on disk.

    #     # Vectorize the data.
    #     input_texts = encoder_input_data
    #     input_len = len(input_texts)
    #     target_texts = []



    #     num_encoder_tokens = len(m.feature_list_full_value)   # remove index
    #     num_decoder_tokens = len(m.test_feature_list)+2
    #     max_encoder_seq_length = 12
    #     # max_decoder_seq_length = 6+1
    #     max_decoder_seq_length = len(decoder_target_data[0])

    #     # Define an input sequence and process it.
    #     encoder_inputs = Input(shape=(None, num_encoder_tokens))
    #     encoder = LSTM(latent_dim, return_state=True)
    #     encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    #     # We discard `encoder_outputs` and only keep the states.
    #     encoder_states = [state_h, state_c]

    #     # Set up the decoder, using `encoder_states` as initial state.
    #     decoder_inputs = Input(shape=(None, num_decoder_tokens))
    #     # We set up our decoder to return full output sequences,
    #     # and to return internal states as well. We don't use the
    #     # return states in the training model, but we will use them in inference.
    #     decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    #     decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    #     decoder_dense = Dense(num_decoder_tokens, activation='sigmoid')
    #     decoder_outputs = decoder_dense(decoder_outputs)


    #     # Define the model that will turn
    #     # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    #     model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    #     # Run training
    #     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])


    #     if FLAGS.train:
    #         model.fit([encoder_input_data, decoder_input_data], decoder_target_data,batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1)
    #         # Save model
    #         model.save(global_path+file_label+'.model')

    #     model = load_model(global_path+file_label+'.model')

    #     encoder_model = Model(encoder_inputs, encoder_states)
    #     decoder_state_input_h = Input(shape=(latent_dim,))
    #     decoder_state_input_c = Input(shape=(latent_dim,))
    #     decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    #     decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    #     decoder_states = [state_h, state_c]
    #     decoder_outputs = decoder_dense(decoder_outputs)
    #     decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    #     if FLAGS.train:
    #         encoder_model.save(global_path+file_label+'.encoder')
    #         decoder_model.save(global_path+file_label+'.decoder')

    #     encoder_model = load_model(global_path+file_label+'.encoder')
    #     decoder_model = load_model(global_path+file_label+'.decoder')


    #     def decode_sequence(input_seq):
    #         # Encode the input as state vectors.
    #         states_value = encoder_model.predict(input_seq)

    #         # Generate empty target sequence of length 1.
    #         target_seq = np.zeros((1, 1, num_decoder_tokens))

    #         # Populate the first character of target sequence with the start character.
    #         # target_seq[0, 0, ] = 1.

    #         # Sampling loop for a batch of sequences
    #         # (to simplify, here we assume a batch of size 1).
    #         stop_condition = False
    #         decoded_sentence = []
    #         while not stop_condition:
    #                 output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        
    #                 pre_tmp_value = output_tokens[0,0,0]

    #                 decoded_sentence.append(pre_tmp_value)
    #                 end_bit = output_tokens[0,0,-1]
            
    #                 # Exit condition: either hit max length
    #                 # or find stop character.

    #                 if (end_bit > 1 or len(decoded_sentence) > max_decoder_seq_length-1):
    #                     stop_condition = True

    #                 # Update the target sequence (of length 1).
    #                 target_seq = np.zeros((1, 1, num_decoder_tokens))
    #                 target_seq[0, 0, 0] = pre_tmp_value
    #                 # Update states

    #                 states_value = [h, c]

    #         return decoded_sentence

    #     label_list = []
    #     pre_list = []

    #     for seq_index in range(len(test_encoder_input_data)):
    #     #for seq_index in range(len(test_encoder_input_data)):
    #         # Take one sequence (part of the training set)
    #         # for trying out decoding.
    #         print(seq_index)
    #         input_seq = test_encoder_input_data[seq_index: seq_index + 1]
        
    #         decoded_sentence = decode_sequence(input_seq)
    #         label_data = [i[0] for i in  test_decoder_target_data[seq_index]]
            
    #         if seq_index < 10:
    #             print('Label sentence:', label_data)
    #             # print('Label sentence:', decoder_target_data[seq_index])
    #             print('Decoded sentence:', decoded_sentence)

    #         label_list.append(label_data)
    #         pre_list.append(decoded_sentence)

    #     print("########################")
    #     m.draw_lstm_without_e(label_list, pre_list, y_scaler, file_label, lstm_path = global_path)



def main(_):
    if FLAGS.task == '1':
        task1()
        
    if FLAGS.task == '2':
        task2()

    if FLAGS.task == '3':
        task3()

    if FLAGS.task == '4':
        task4()

    if FLAGS.task == '5':
        task5()

    if FLAGS.task == '6':
        task6()

    if FLAGS.task == '7':
        task7()

    if FLAGS.task == '8':
        task8()

    if FLAGS.task == '9':
        task9()
    
    if FLAGS.task == '10':
        task10()
    
    if FLAGS.task == '11':
        task11()

    if FLAGS.task == '12':
        task12()

    if FLAGS.task == '13':
        task13()

    if FLAGS.task == '14':
        task14()

    if FLAGS.task == '15':
        task15()

    if FLAGS.task == '16':
        task16()
    
    if FLAGS.task == '17':
        task17()

    if FLAGS.task == '18':
        task18()



if __name__ == '__main__':
    app.run(main)