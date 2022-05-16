#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import os

import tensorflow as tf

from utils import *
from preprecoess import *
from generator import *
from models import *
from fl_training import *

import numpy as np



root_data_path = "./EduFL/data/MoocData/"   # MoocData Path
embedding_path = "./EduFL/data/glove/" # MoocData Word Embedding Path
results = "./EduFL/Results/MoocData/"



# Read items
items,items_index,category_dict,subcategory_dict,word_dict,cate_subcate_dict = read_news(root_data_path,['train','val'])
items_title,items_vert,items_subvert = get_doc_input(items, items_index, category_dict, subcategory_dict, word_dict)

title_word_embedding_matrix, have_word = load_matrix(embedding_path,word_dict)
items_embedding, cate_Embedding = represent_items(embedding_path, items, items_title, items_index, word_dict, title_word_embedding_matrix, category_dict, subcategory_dict, cate_subcate_dict)

#Parse User
train_session, train_uid_click, train_uid_table, train_user_list = read_clickhistory(root_data_path,'train')

test_session, test_uid_click,test_uid_table, test_user_list = read_clickhistory(root_data_path,'val')
train_user = parse_user(train_session,items_index)
test_user = parse_user(test_session,items_index)
train_sess, train_user_id, train_label, train_user_id_sample = get_train_input(train_session,train_uid_click,items_index)
test_impressions, test_userids = get_test_input(test_session,items_index)

mu, user_list, lenclick_user_dict, lenclick_frequency = compute_lenclick(root_data_path,['train','val'])
# print(lenclick_frequency) 
                                                     

len_click_pmf, len_click_pmf_user_dict = compute_pmf(lenclick_user_dict, lenclick_frequency, user_list, train_uid_table, train_user_list)


get_user_data = GetUserDataFunc(items_title,train_user_id_sample,train_user,train_sess,train_label,train_user_id, items_embedding)

lr = 0.40   # Mooc:trans  performence:0.4>0.2
delta = 0.05
lambd = 0
ratio = 0.02
num = 50  
alpha = 0.5   # Mooc:performence:0.5>0.8,It is useful to consider different types of categary.
alpha = 0.5  # Mooc:performence:0.5>0.2 & 0.2>0.1,the weight of categary should not too large.  
beta = 0.5  # Mooc:performence:0.5>0.1

model, user_encoder = get_model(lr,delta,title_word_embedding_matrix, cate_Embedding, alpha)
print(model.summary())

Res = []
Loss = []
count = 0

while count<=10000:
    loss = fed_single_update(model, user_encoder, num, lambd, get_user_data, train_uid_table, items_embedding, len_click_pmf_user_dict, len_click_pmf, beta)   
    Loss.append(loss)
    with open(results + 'Loss.json','a') as f1:   
        s = json.dumps(loss) + '\n'
        f1.write(s)
    print(loss)
    if count % 10 == 0:  
        items_scoring = items_embedding
        user_generator = get_hir_user_generator(items_scoring,test_user['click'],64)   
        user_scoring = user_encoder.predict_generator(user_generator,verbose=0)  
        g = evaluate(user_scoring,items_scoring,test_impressions)
        Res.append(g)
        print(g)
        with open(results + 'performance.json','a') as f:
            s = json.dumps(g) + '\n'
            f.write(s)
    count += 1

    