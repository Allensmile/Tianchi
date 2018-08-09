from keras.layers import InputSpec, Layer, Input, Dense, merge, Conv1D, Dot, Permute, Multiply
from keras.layers import Lambda, Activation, Dropout, Embedding, TimeDistributed
from keras.layers.core import SpatialDropout1D
from keras.layers.recurrent import LSTM, GRU
from keras.layers.pooling import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import Bidirectional
import keras.backend as K
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.optimizers import Nadam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
import numpy as np 
import math
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint

# paper : Enhanced LSTM Attention model by Qian Chen et al. 2016

class ELA:
    @staticmethod
    def build_model(emb_matrix, max_sequence_length):
        ############# Embedding Process ############
        # The embedding layer containing the word vectors
        emb_layer = Embedding(
            input_dim=emb_matrix.shape[0],
            output_dim=emb_matrix.shape[1],
            weights=[emb_matrix],
            input_length=max_sequence_length,
            trainable=False
        )
        # Enhanced attention model ##########
        # Define inputs
        seq1 = Input(shape=(max_sequence_length,))
        seq2 = Input(shape=(max_sequence_length,))

        # Run inputs through embedding
        emb1 = emb_layer(seq1)
        emb2 = emb_layer(seq2)
        
        # The encoder layer
        # encode words with its surrouding context
        encode_layer = Bidirectional(LSTM(units = 300, return_sequences = True))
        emb1 = Dropout(0.4)(encode_layer(emb1))
        emb2 = Dropout(0.4)(encode_layer(emb2))
        
        # score each words and calculate score matrix
        cross = Dot(axes = (2,2))([emb1,emb2])
        c1 = Lambda(lambda x: keras.activations.softmax(x))(cross)
        c2 = Permute((2,1))(cross)
        c2 = Lambda(lambda x: keras.activations.softmax(x))(c2)
        
        # normalize score matrix, encoder premesis and get alignment
        seq1Align = Dot((2,1))([c1,emb2])
        seq2Align = Dot((2,1))([c2,emb1])
        mm_1 = Multiply()([emb1,seq1Align])
        mm_2 = Multiply()([emb2,seq2Align])
        sb_1 = Lambda(lambda x: tf.subtract(x, seq1Align))(emb1)
        sb_2 = Lambda(lambda x: tf.subtract(x, seq2Align))(emb2)
        
        seq1Align = concatenate([emb1,seq1Align,mm_1,sb_1])
        seq2Align = concatenate([emb2,seq2Align,mm_2,sb_2])
        seq1Align = Dropout(0.4)(seq1Align)
        seq2Align = Dropout(0.4)(seq2Align)
        
        compresser = TimeDistributed(Dense(300,
                                          kernel_regularizer=l2(1e-5),
                                          bias_regularizer=l2(1e-5),
                                          activation='relu')
                                    )

        seq1Align = compresser(seq1Align)
        seq2Align = compresser(seq2Align)
        
        # biLSTM  Decoder
        decode_layer = Bidirectional(LSTM(units=300, return_sequences = True))
        final_seq1 = Dropout(0.4)(decode_layer(seq1Align))
        final_seq2 = Dropout(0.4)(decode_layer(seq2Align))
        
        averagePooling = Lambda(lambda x: K.mean(x, axis=1))
        maxPooling = Lambda(lambda x: K.max(x, axis=1))
        avg_seq1 = averagePooling(final_seq1)
        avg_seq2 = averagePooling(final_seq2)
        max_seq1 = maxPooling(final_seq1)
        max_seq2 = maxPooling(final_seq2)
        
        merged = concatenate([avg_seq1,max_seq1,avg_seq2,max_seq2])
        merged = Dropout(0.4)(merged)
        
        merged = Dense(300,
                       kernel_regularizer=l2(1e-5),
                       bias_regularizer=l2(1e-5),
                       activation='relu')(merged)
        merged = Dropout(0.2)(merged)
        merged = BatchNormalization()(merged)

        pred = Dense(1, activation='sigmoid')(merged)

        # model = Model(inputs=[seq1, seq2, magic_input, distance_input], outputs=pred)
        model = Model(inputs=[seq1, seq2], outputs=pred)
        model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['acc'])
        print (model.summary())
        return model