import numpy
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Embedding, Concatenate    

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras import initializers #keras2
from tensorflow.keras.utils import plot_model
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.optimizers import *

import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import InputLayer, Dense, Embedding, Reshape, BatchNormalization, Masking,Lambda  # my code!

from trans import *

npratio = 4
ls2t_size = 100
ls2t_order = 3
ls2t_depth = 3


class Attention(Layer):
 
    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)
 
    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)
 

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12
 
    def call(self, x):
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        
        Q_seq = K.dot(Q_seq, self.WQ) 
        # print('Q_seq', Q_seq.shape)   
        Q_seq = K.reshape(Q_seq, (-1, Q_seq.shape[1], self.nb_head, self.size_per_head)) 
        # print('Q_seq', Q_seq.shape)  
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        
        K_seq = K.dot(K_seq, self.WK)   
        K_seq = K.reshape(K_seq, (-1, K_seq.shape[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))  
        
        V_seq = K.dot(V_seq, self.WV)  
        V_seq = K.reshape(V_seq, (-1, V_seq.shape[1], self.nb_head, self.size_per_head)) 
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))  
        # print('V_seq',V_seq.shape)   

        A = tf.matmul(Q_seq, K.permute_dimensions(K_seq, (0,1,3,2))) / self.size_per_head**0.5  
        # print('A',A.shape)  
        
        A = K.permute_dimensions(A, (0,3,2,1))   
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))   
        A = K.softmax(A)
        # print('A',A.shape)  

        O_seq = K.batch_dot(A, V_seq, axes=[3,2])   
        O_seq = tf.matmul(A, V_seq)   
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, O_seq.shape[1], self.output_dim))  
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        # print('O_seq', O_seq.shape) 
        return O_seq
 
    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


def AttentivePooling(dim1,dim2):
    vecs_input = Input(shape=(dim1,dim2),dtype='float32')  
    user_vecs =Dropout(0.2)(vecs_input)    
    user_att = Dense(150,activation='tanh')(user_vecs)   
    user_att = tf.keras.layers.Flatten()(Dense(1)(user_att))
    user_att = Activation('softmax')(user_att)
    user_vec = tf.keras.layers.Dot((1,1))([user_vecs,user_att])
    model = Model(vecs_input,user_vec)
    return model


def get_user_encoder(cate_Embedding, alpha):   
    vecs_input = Input(shape=(60,300), dtype='float32')
    translayer1 = EncoderLayer(d_model=300, n_heads=20, ddf=600)(vecs_input)
    translayer2 = EncoderLayer(d_model=300, n_heads=20, ddf=600)(translayer1)
    translayer3 = EncoderLayer(d_model=300, n_heads=20, ddf=600)(translayer2)
    MicroPrefer = Lambda(lambda x:x[:,-1,:])(translayer3)   
    # print(MicroPrefer.shape) 
    
    MicroPreferr = tf.cast(MicroPrefer,dtype=tf.float64)
    MicroPrefer_tmp = tf.expand_dims(MicroPreferr, axis = 1)
    # print(MicroPrefer_tmp.shape) 
    
    cate_Em = tf.convert_to_tensor(cate_Embedding)  
    MicroPr_cate = MicroPrefer_tmp * cate_Em
    # print(MicroPr_cate.shape) 
    MicroPr_cate_sum = tf.reduce_sum(MicroPr_cate, axis=-1)
    att = MicroPr_cate_sum / tf.reduce_sum(MicroPr_cate_sum)
    att = tf.expand_dims(att, axis = 1)   
    MacroPrefer = tf.squeeze(tf.matmul(att, cate_Em), axis=1)  
       
    vec = alpha*MicroPreferr + (1-alpha)*MacroPrefer  
     
      
    sentEncodert = Model(inputs=vecs_input, outputs = vec)
    return sentEncodert   

    

def get_model(lr,delta,title_word_embedding_matrix, cate_Embedding, alpha):
    user_encoder = get_user_encoder(cate_Embedding, alpha)   # transEncoder
 
    click_vecs = Input(shape=(60,300),dtype='float32')   
    can_vecs = Input(shape=(60,300),dtype='float32') 
    
    user_vec = user_encoder(click_vecs)
    # print('user_vec',user_vec.shape)
    
    scores = tf.keras.layers.Dot(axes=-1)([user_vec,can_vecs]) #(batch_size,1+1,) 
    logits = tf.keras.layers.Activation(tf.keras.activations.softmax,name = 'recommend')(scores)     
    
    model = Model([can_vecs,click_vecs],logits) # max prob_click_positive

    model.compile(loss=['categorical_crossentropy'],    
                  optimizer=SGD(lr=lr,clipvalue = delta), 
                  metrics=['acc'])   

    return model, user_encoder