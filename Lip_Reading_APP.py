import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import streamlit as st


def load_video(path:str) -> List[float]:

    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236,80:220,:])
    cap.release()

    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std


vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]


char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
num_to_char = tf.keras.layers.StringLookup(
    vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True
)



def load_alignments(path:str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens = [*tokens,' ',line[2]]
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]


def load_data(path: str):
    path = bytes.decode(path.numpy())
    # file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join('data','s1',f'{file_name}.mpg')
    alignment_path = os.path.join('data','alignments','s1',f'{file_name}.align')
    frames = load_video(video_path)
    alignments = load_alignments(alignment_path)

    return frames, alignments



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten

def load_model() -> Sequential:
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

    model.load_weights(r'C:\Users\ASUS\OneDrive\Desktop\Lip Reading\models - checkpoint 50\models\checkpoint')
    return model







st.set_page_config(layout='wide')


st.title('Lip Reading Model') 
options = os.listdir(r'C:\Users\ASUS\OneDrive\Desktop\Lip Reading\data\s1')

col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        vid1 = st.text_input('enter video name')
        vid = vid1 + '.mpg'
        
        if vid == '.mpg':
            st.info('The video below displays the converted video in mp4 format')
            file_path = os.path.join(r'C:\Users\ASUS\OneDrive\Desktop\Lip Reading\data\s1', 'bgan4n.mpg')
            os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
        else:
            st.info('The video below displays the converted video in mp4 format')
            file_path = os.path.join(r'C:\Users\ASUS\OneDrive\Desktop\Lip Reading\data\s1', vid)
            os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
                     
        vif = vid1 + '.mp4'
        if vif == '.mp4':
            

            gg = os.path.join(r"C:\Users\ASUS\OneDrive\Desktop\Lip Reading\test videos" , "bgan4n.mp4")

            video = open(gg, 'rb') 
            video_bytes = video.read() 
            st.video(video_bytes)
        else:
            gg = os.path.join(r"C:\Users\ASUS\OneDrive\Desktop\Lip Reading\test videos" , vif)
            video = open(gg, 'rb') 
            video_bytes = video.read() 
            st.video(video_bytes)


    with col2: 
        video , ggggg = load_data(tf.convert_to_tensor(file_path))
        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)


        st.info('Predicted Words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
        st.info('Real Words')
        sample = load_data(tf.convert_to_tensor(file_path))
        f = [tf.strings.reduce_join([num_to_char(word) for word in sentence]) for sentence in [sample[1]]]
        st.text(f)
