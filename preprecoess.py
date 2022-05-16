import csv
import datetime
import time
import json
import itertools
import random
import os
import numpy as np
from tensorflow.keras.layers import Embedding
import math


MAX_SENTENCE = 10
MAX_ALL = 60
npratio = 4

def newsample(nnn,ratio):
    if ratio >len(nnn):
        return random.sample(nnn*(ratio//len(nnn)+1),ratio)
    else:
        return random.sample(nnn,ratio)
        

def read_news(root_data_path, modes):
    items={}
    category=[]
    subcategory=[]
    items_index={}
    index=1
    word_dict={}
    word_index=1
    cate_subcate_tmp = []
    
    for mode in modes:   # Put the train set and val set together
        with open(os.path.join(root_data_path,mode,'items.tsv')) as f:    
            lines = f.readlines()
        for line in lines:
            splited = line.strip('\n').split('\t')
            tmp = {}
            doc_id = splited[0]   # MoocData
            vert = splited[1].split(' ')[0]   # MoocData
            subvert = splited[1].split(' ')[1]   # MoocData
            tmp[vert] = subvert  # MoocData
            cate_subcate_tmp.append(tmp)  # MoocData
            title = splited[2]   # MoocData
                
            if doc_id in items_index:
                continue
            items_index[doc_id]=index  # reate a dictionary for news,key:doc_id,value:index
            index+=1
            category.append(vert)
            subcategory.append(subvert)
            
            title = title.split(' ')

            items[doc_id]=[vert,subvert,title]
            # title = title.split(' ')
            for word in title:               
                if not(word in word_dict):
                    word_dict[word]=word_index
                    word_index+=1
    category=list(set(category))
    subcategory=list(set(subcategory))
    category_dict={}
    index_category = 1
    for c in category:
        category_dict[c] = index_category
        index_category += 1
    subcategory_dict={}
    index_subcategory = 1
    for c in subcategory:
        subcategory_dict[c] = index_subcategory
        index_subcategory += 1
    
    cate_subcate_dict = {}
    for _ in cate_subcate_tmp:
        for k, v in _.items():
            cate_subcate_dict.setdefault(k, []).append(v)
    
    return items,items_index,category_dict,subcategory_dict,word_dict,cate_subcate_dict


def get_doc_input(items,items_index,category,subcategory,word_dict):
    items_num = len(items)+1  
    items_title = np.zeros((items_num, MAX_SENTENCE),dtype='int32')
    items_vert = np.zeros((items_num,),dtype='int32')
    items_subvert = np.zeros((items_num,),dtype='int32')
    for key in items:    
        vert,subvert,title = items[key]
        doc_index = items_index[key]
        items_vert[doc_index] = category[vert]
        items_subvert[doc_index]=subcategory[subvert]
        for word_id in range(min(MAX_SENTENCE,len(title))):
            items_title[doc_index,word_id] = word_dict[title[word_id]]  
    return items_title,items_vert,items_subvert


def load_matrix(embedding_path,word_dict):  
    embedding_matrix_Em = Embedding(len(word_dict)+1, 300, trainable=False)
    embedding_matrix0 = np.zeros((len(word_dict)+1,300))
    for i in range(len(word_dict)+1):
        embedding_matrix0[i] = embedding_matrix_Em(i)
    embedding_matrix1 = np.zeros((len(word_dict)+1,300))
    have_word=[]
    all_word = [k for k in word_dict]
    
    with open(os.path.join(embedding_path,'glove.baidubaike.bigram.txt'),'rb') as f:
        while True:
            l=f.readline()
            if len(l)==0:
                break
            l=l.split()
            word = l[0].decode()
            if word in word_dict:
                index = word_dict[word]
                tp = [float(x) for x in l[1:]]
                embedding_matrix1[index]=np.array(tp)
                have_word.append(word)
            if word in list(set(all_word).difference(set(have_word))):
                index = word_dict[word]
                embedding_matrix1[index]=embedding_matrix0[index]     
    embedding_matrix = (embedding_matrix0 + embedding_matrix1)/2            
    return embedding_matrix,have_word

def represent_items(embedding_path, items, items_title, items_index, word_dict, title_word_embedding_matrix, category_dict, subcategory_dict, cate_subcate_dict):
    items_dict = {}
    for k,v in items_index.items():
        items_dict[v] = k
    
    tmp = {}
    with open(os.path.join(embedding_path,'glove.baidubaike.bigram.txt'),'rb') as f:
        while True:
            l=f.readline()
            if len(l)==0:
                break
            l=l.split()
            word = l[0].decode()
            tp = [float(x) for x in l[1:]]
            tmp[word]=np.array(tp)
        
    cate_Embedding = []
    cate_dict = {}
    for k,v in category_dict.items():
        cate_Em = []
        cate_dict[v] = k
        # print(k)
        k_emb1 = tmp[k]
        cate_Embedding.append(k_emb1)
        
    subcate_dict = {}
    for k,v in subcategory_dict.items():
        subcate_dict[v] = k   
    
    sub_Embedding = []
    items_Embedding = np.zeros((len(items_index)+1,300))
    for k,v in items_index.items():        
        EmI = []
        subcate = items[k][1]
        # print([subcate][0])
        EmI1 = tmp[subcate]
        EmI.append(EmI1)        
        tmpp = []
        title = items_title[v]
        for j in range(len(title)):
            wordID = title[j]
            wordEm = title_word_embedding_matrix[wordID]
            tmpp.append(wordEm)
        EmI2 = np.mean(np.array(tmpp), axis=0)
        EmI.append(EmI2)
        Em_I = np.mean(np.array(EmI), axis=0)
        items_Embedding[v] = EmI2
    return np.array(items_Embedding),np.array(cate_Embedding)


def read_clickhistory(root_data_path,mode):
    
    user_list = []
    lines = []
    userids = {}
    uid_table = {}
    with open(os.path.join(root_data_path,mode,'behaviors.tsv')) as f:
        lines = f.readlines()
        
    sessions = []
    for i in range(len(lines)):
        _,uid,_,click,imp = lines[i].strip().split('\t')
        user_list.append(uid)
        true_click = click.split()
        assert not '' in true_click
        if not uid in userids:
            uid_table[len(userids)] = uid
            userids[uid] = []  # A user has more than one record
        userids[uid].append(i)
        imp = imp.split()    
        pos = []
        neg = []
        for beh in imp:
            nid, label = beh.split('+++')    
            if label == '0':
                neg.append(nid)
            else:
                pos.append(nid)
        sessions.append([true_click,pos,neg])
    return sessions,userids,uid_table,user_list


def parse_user(session,items_index):
    user_num = len(session)
    user={'click': np.zeros((user_num,MAX_ALL),dtype='int32'),} 
    for user_id in range(len(session)):
        tclick = []
        click, pos, neg = session[user_id]
        for i in range(len(click)):
            tclick.append(items_index[click[i]])
        click = tclick

        if len(click) > MAX_ALL:
            click = click[-MAX_ALL:]
        else:
            click=[0]*(MAX_ALL-len(click)) + click
            
        user['click'][user_id] = np.array(click)
    return user


def get_train_input(session,uid_click_talbe,items_index):
    inv_table = {}
    user_id_session = {}

    for uid in uid_click_talbe:
        user_id_session[uid] = []
        for v in uid_click_talbe[uid]:
            inv_table[v] = uid
    
    sess_pos = []
    sess_neg = []
    user_id = []
    for sess_id in range(len(session)):
        sess = session[sess_id]
        _, poss, negs=sess
        for i in range(len(poss)):
            pos = poss[i]
            neg=newsample(negs,npratio)
            sess_pos.append(pos)
            sess_neg.append(neg)
            user_id.append(sess_id)                
            user_id_session[inv_table[sess_id]].append(len(sess_pos)-1)
            
    sess_all = np.zeros((len(sess_pos),1+npratio),dtype='int32')
    label = np.zeros((len(sess_pos),1+npratio))
    for sess_id in range(sess_all.shape[0]):
        pos = sess_pos[sess_id]
        negs = sess_neg[sess_id]
        sess_all[sess_id,0] = items_index[pos]
        index = 1
        for neg in negs:
            sess_all[sess_id,index] = items_index[neg]
            index+=1
        label[sess_id,0]=1
    user_id = np.array(user_id, dtype='int32')
    # print(user_id_session)
    return sess_all, user_id, label, user_id_session


def get_test_input(session,items_index):
    
    Impressions = []
    userid = []
    for sess_id in range(len(session)):
        _, poss, negs = session[sess_id]
        imp = {'labels':[],
                'docs':[]}
        userid.append(sess_id)
        for i in range(len(poss)):
            docid = items_index[poss[i]]
            imp['docs'].append(docid)
            imp['labels'].append(1)
        for i in range(len(negs)):
            docid = items_index[negs[i]]
            imp['docs'].append(docid)
            imp['labels'].append(0)
        Impressions.append(imp)
        
    userid = np.array(userid,dtype='int32')
    # print(userid)
    
    return Impressions, userid
    
       
def compute_lenclick(root_data_path,modes):
    user_list = []
    len_sum = []
    user_lenclick_list = []
    for mode in modes:   # Put the train set and val set together
        with open(os.path.join(root_data_path,mode,'behaviors.tsv')) as f:
            lines = f.readlines()
        
           
        for i in range(len(lines)):          
            _,uid,_,click,imp = lines[i].strip().split('\t')
            user_list.append(uid)
            user_lenclick_dict_tmp = {} 
            true_click = click.split(',')
            user_lenclick_dict_tmp[uid] = len(true_click)
            user_lenclick_list.append(user_lenclick_dict_tmp) 
            len_sum.append(len(true_click))          
    len_sum = np.array(len_sum).sum()
    mu = math.floor(len_sum/len(user_list))
            
    lenclick_user_list = []   
    for i in range(len(user_lenclick_list)):
        lenclick_user_list_tmp = {}        
        for k,v in user_lenclick_list[i].items():
            lenclick_user_list_tmp[v] = k
            lenclick_user_list.append(lenclick_user_list_tmp)
            
    lenclick_user_dict = {}
    for _ in lenclick_user_list:
        for k, v in _.items():
            lenclick_user_dict.setdefault(k, []).append(v)
    # print(lenclick_user_dict[22])   
            
    lenclick_frequency = {}    
    for k, v in lenclick_user_dict.items():
        lenclick_frequency[k] = len(v)   
    # lenclick_frequency = sorted(lenclick_frequency.items(),key=lambda x:x[1],reverse=False)
    # print(lenclick_frequency)
    # {11: 1033, 7: 1738, 13: 870, 27: 321, 25: 331, 9: 1350, 45: 127, 68: 48, 5: 2318, 51: 102, 8: 1433, 43: 158, 6: 1989, 10: 1163, 38: 155, 15: 676, 19: 509, 12: 956, 16: 624, 48: 109, 102: 18, 42: 169, 47: 129, 17: 602, 14: 765, 26: 338, 30: 250, 90: 22, 85: 31, 55: 91, 64: 44, 94: 21, 66: 67, 28: 300, 24: 373, 32: 250, 23: 385, 20: 500, 21: 475, 33: 228, 4: 122, 44: 161, 22: 371, 49: 121, 75: 35, 29: 242, 80: 41, 126: 13, 46: 115, 70: 41, 61: 83, 40: 158, 72: 38, 39: 168, 52: 88, 107: 15, 36: 198, 35: 204, 31: 267, 63: 63, 65: 59, 205: 2, 18: 539, 41: 160, 34: 215, 188: 8, 60: 93, 69: 44, 57: 69, 137: 9, 50: 116, 76: 46, 67: 45, 125: 13, 122: 8, 56: 71, 37: 198, 83: 30, 196: 3, 59: 65, 62: 63, 53: 87, 58: 55, 160: 7, 79: 42, 144: 3, 148: 7, 95: 18, 73: 34, 140: 6, 116: 12, 216: 2, 336: 1, 92: 23, 88: 18, 78: 36, 81: 29, 112: 11, 87: 21, 119: 9, 82: 32, 77: 40, 86: 24, 103: 16, 223: 2, 71: 58, 54: 93, 175: 3, 165: 4, 104: 13, 121: 4, 127: 4, 214: 1, 136: 5, 98: 14, 252: 1, 209: 2, 194: 2, 129: 7, 97: 28, 302: 1, 91: 26, 279: 1, 109: 12, 84: 15, 130: 4, 155: 4, 74: 41, 120: 8, 110: 12, 304: 1, 128: 6, 221: 1, 96: 22, 199: 1, 117: 6, 111: 7, 118: 9, 131: 12, 101: 10, 115: 11, 139: 4, 113: 7, 162: 3, 106: 13, 142: 11, 93: 15, 105: 12, 89: 20, 517: 1, 468: 1, 99: 11, 100: 14, 328: 1, 187: 1, 134: 4, 203: 2, 178: 4, 123: 8, 293: 2, 314: 1, 114: 9, 265: 2, 143: 6, 138: 2, 141: 2, 108: 21, 197: 1, 147: 4, 292: 1, 309: 1, 166: 4, 243: 1, 222: 2, 201: 2, 156: 1, 208: 3, 317: 1, 151: 7, 164: 4, 135: 8, 352: 1, 158: 2, 153: 3, 146: 3, 145: 2, 227: 1, 233: 1, 132: 5, 174: 3, 213: 3, 204: 1, 225: 1, 169: 2, 190: 1, 163: 1, 256: 1, 173: 3, 167: 2, 124: 8, 239: 1, 359: 1, 294: 2, 245: 1, 296: 1, 299: 1, 234: 1, 253: 2, 325: 1, 184: 1, 379: 1, 263: 1, 133: 5, 180: 3, 192: 3, 159: 4, 170: 2, 150: 1, 360: 1, 176: 1, 311: 1, 242: 1, 264: 1, 149: 3, 152: 1, 219: 1, 211: 1, 277: 1, 237: 1, 189: 1, 157: 3, 186: 2, 385: 1, 154: 1, 181: 2, 250: 1, 261: 2, 177: 1, 202: 1, 323: 1, 228: 1, 427: 1, 193: 2, 183: 2, 168: 1, 845: 1, 172: 1, 248: 1, 171: 1, 249: 1, 185: 1, 200: 1, 179: 1, 285: 1}

    return mu, user_list, lenclick_user_dict, lenclick_frequency
    

def compute_pmf(lenclick_user_dict, lenclick_frequency, user_list, train_uid_table, train_user_list):
                
    len_click_pmf = {}
    len_click_pmf_user_dict = {}
    train_uid_table_tmp = {}
    for k, v in train_uid_table.items():  
        train_uid_table_tmp[v] = k
    len_click_pmf_user_dict = {}
    for k, v in lenclick_frequency.items():
        len_click_pmf[k] = v / len(user_list)
        tmd = []
        for i in range(len(lenclick_user_dict[k])):    
            if lenclick_user_dict[k][i] in train_user_list:
                userID = lenclick_user_dict[k][i]
                tmd.append(train_uid_table_tmp[userID])
            len_click_pmf_user_dict[k] = tmd  
        
    return len_click_pmf, len_click_pmf_user_dict

        
