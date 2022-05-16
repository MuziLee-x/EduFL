import numpy as np
import tensorflow as tf
import math
import random


def GetUserDataFunc(items_title,train_user_id_sample,train_user,train_sess,train_label,train_user_id, items_embedding):
    def _get_user_data(uid):  
        click = []
        sample = []
        label = []
        for sid in train_user_id_sample[uid]:
            click.append(train_user['click'][train_user_id[sid]])
            sample.append(train_sess[sid])
            label.append(train_label[sid])       
        click = np.array(click)
        sample = np.array(sample)
        label = np.array(label)
        click = items_embedding[click]
        sample = items_embedding[sample]      
        return click,sample,label
    return _get_user_data


def add_noise(weights,lambd):
    for i in range(len(weights)):
        weights[i] += np.random.laplace(scale = lambd,size=weights[i].shape)
    return weights
    

def fed_single_update(model, user_encoder, num, lambd, get_user_data, train_uid_table, news_embedding, len_click_pmf_user_dict, len_click_pmf, beta):
    
    len_click_pmf_keys = list(len_click_pmf.keys())
    len_click_pmf_values = list(len_click_pmf.values())
    # print(len_click_pmf_values)
    
    random_index = []
    w = []
    samples_len_click = tf.random.categorical(tf.math.log([len_click_pmf_values]), num)
    samples_len_click_tmp = samples_len_click.numpy()[0].tolist()  
    # print(samples_len_click_tmp)   # [17, 3, 0, 43, 12, 8, 15, 17, 1, 8, 79, 1, 16, 2, 2, 17, 26, 72, 60, 34,...]
    for i in range(len(samples_len_click_tmp)):
        len_click_keys = len_click_pmf_keys[samples_len_click_tmp[i]]
        if len(len_click_pmf_user_dict[len_click_keys])>1:
            userOnehot = random.sample(len_click_pmf_user_dict[len_click_keys], 1)
            random_index.append(userOnehot[0])
            w.append(len_click_pmf_values[samples_len_click_tmp[i]]) 
    
    random_index = np.array(random_index)
    w = np.array(w)
    w = w/w.sum()
    
    all_user_weights = []
    old_user_weight = user_encoder.get_weights()    
    sample_nums = []

    loss = []
    for uinx in random_index:
        user_encoder.set_weights(old_user_weight)

        uid = train_uid_table[uinx]
        click,sample,label = get_user_data(uid)
        g = model.fit([sample,click],label,batch_size = label.shape[0],verbose=False)
        loss.append(g.history['loss'][0])
        user_weight = user_encoder.get_weights()
        if lambd>0:
            news_weight = add_noise(news_weight,lambd)
            user_weight = add_noise(user_weight,lambd)

        all_user_weights.append(user_weight)
        sample_nums.append(label.shape[0])
    
    sample_nums = np.array(sample_nums)
    sample_nums = sample_nums/sample_nums.sum()
    
    sample_nums_revised = beta*w + (1-beta)*sample_nums
    
    user_weights = [np.average(weights, axis=0,weights=sample_nums_revised) for weights in zip(*all_user_weights)]
    
    user_encoder.set_weights(user_weights)
    
    loss = np.array(loss).mean()

    return loss