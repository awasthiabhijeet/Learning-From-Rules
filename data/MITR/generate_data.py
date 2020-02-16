#Global variables
from get_elmo_contextual_embeddings import sentences_to_contx_embs
from rules import *

root_path='data_files/'
file_names= ['d','U','validation','test']
d_flag = [1,0,0,0]
u_flag = [0,1,0,0]

d_size=185
v_size=500

x=[]
l=[]
m=[]
L=[]
d=[]
r=[]

num_classes=9

rules_center_word_file_path=root_path+'rules_center.words.txt'
rules_center_tag_file_path=root_path+'rules_center.tags.txt'

all_word_file_path=root_path+'all.words.txt'
all_tag_file_path=root_path+'all.tags.txt'

import numpy as np
import pickle
import random

class2label = {'O': 0,
                 'Location': 1, 'Hours': 2,
                 'Amenity': 3, 'Price': 4,
                 'Cuisine': 5, 'Dish': 6,
                 'Restaurant_Name': 7, 'Rating': 8}

label2class = {0: 'O',
               1: 'Location', 2: 'Hours',
               3: 'Amenity', 4: 'Price',
               5: 'Cuisine', 6: 'Dish',
               7: 'Restaurant_Name', 8: 'Rating'}



#utils
def get_sent_dict(sentence):
  list_of_words=sentence.strip().split(" ")
  length_of_words=[len(i)+1 for i in list_of_words]
  words=[0]
  words.extend(length_of_words[0:-1])
  cummulative_list=np.cumsum(words)
  cummulative_list[0]=0
  sent_dict=dict(zip(cummulative_list,range(len(cummulative_list))))
  return sent_dict

def update_lm(l,m,firing_rule,rule_id):
  for i in range(0,len(firing_rule)):
    if firing_rule[i]!=0:
      l[i][rule_id]=firing_rule[i]
      m[i][rule_id]=1
  return l,m

#lists functions

def clear_lists():
    x.clear()
    l.clear()
    m.clear()
    L.clear()
    d.clear()
    r.clear()

def print_list():
    print(len(l))

def dump_lists_in_pickle(output_processed_file_path):
    processed_file=open(output_processed_file_path,'wb')
    #print("dump_length",len(x)+len(l)+len(m)+len(L)+len(d)+len(r))
    print("dump_length",len(x))
    xarray=np.array(x)
    larray=np.array(l)
    marray=np.array(m)
    Larray=np.array(L)
    darray=np.array(d)
    rarray=np.array(r)
    pickle.dump(xarray,processed_file)
    pickle.dump(larray,processed_file)
    pickle.dump(marray,processed_file)
    pickle.dump(Larray,processed_file)
    pickle.dump(darray,processed_file)
    pickle.dump(rarray,processed_file)
    processed_file.close()
    clear_lists()

def fill_lists(word_file_path, tag_file_path, d_flag, r_flag, u_flag):
    with open(word_file_path,'r') as word_file,\
        open(tag_file_path,'r') as tag_file:
        idx=0
        total_words=0
        sentences = []
        for line,tags in zip(word_file,tag_file):
            sentences.append(line.strip())
            idx=idx+1
            #List of words 
            words=line.strip().split(" ")
            num_of_words=len(words)
            total_words=total_words+num_of_words
            x_temp=[[0]*1024 for i in range(0,num_of_words)]

            sent_dict=get_sent_dict(line)
            
            l_temp=[[num_classes]*num_rules for i in range(0,num_of_words)]
            m_temp=[[0]*num_rules for i in range(0,num_of_words)]
            r_temp=[[0]*num_rules for i in range(0,num_of_words)]
            
            for rule,rule_id in zip(rule_list,range(0,num_rules)):
                firing_rule=[0]*len(sent_dict.keys())
                firing_rule=rule(line,sent_dict,firing_rule)
                l_temp,m_temp=update_lm(l_temp,m_temp,firing_rule,rule_id)
                if r_flag==1 and rule_id==idx-1:
                    for i in range(0,len(firing_rule)):
                       if(firing_rule[i]!=0):
                          r_temp[i][rule_id]=1
                    #print("firing_rule : ",firing_rule)
                    #print("r_temp : ",r_temp)
                    #print("##############")

            #L: gold labels
            true_labels=tags.strip().split(" ")

            if u_flag==1:
                L_temp=[num_classes for i in range(0,num_of_words)]
            else:
                L_temp=[class2label[i] for i in true_labels]
            
            if d_flag ==1:
                d_temp=[1 for i in range(0,num_of_words)]
            else:
                d_temp=[0 for i in range(0,num_of_words)]
            l.extend(l_temp)
            m.extend(m_temp)
            L.extend(L_temp)
            d.extend(d_temp)
            r.extend(r_temp)
        global x
        embeddings = sentences_to_contx_embs(sentences)
        embeddings = [item for sublist in embeddings for item in sublist]
        x.extend(embeddings)
        print("total_words",total_words)

def dump_list_to_txt(file_path,list_data):
    with open(file_path,'w') as f:
        for line in list_data:
            f.write(line.strip()+"\n")
    
def generate_word_tag_files():

    with open(all_word_file_path,'r') as word_file,\
        open(all_tag_file_path,'r') as tag_file,\
        open(rules_center_word_file_path,'r') as rule_word_file,\
        open(rules_center_tag_file_path,'r') as rule_tag_file:

        words = word_file.read().strip().split('\n')
        tags = tag_file.read().strip().split('\n')

        rule_words = rule_word_file.read().strip().split('\n')
        rule_tags = rule_tag_file.read().strip().split('\n')

        words_tags = list(zip(words, tags))
        random.shuffle(words_tags)
        words,tags = zip(*words_tags)

        d_words = words[0:d_size]+tuple(rule_words)
        d_tags = tags[0:d_size]+tuple(rule_tags)
        v_words = words[-v_size:]
        v_tags = tags[-v_size:]
        U_words = words[d_size:-v_size]
        U_tags = tags[d_size:-v_size]
        print(type(d_words))

        f_names=['d','validation','U']
        w=[d_words,v_words,U_words]
        t=[d_tags,v_tags,U_tags]

        for fname,words_list,tags_list in zip(f_names,w,t):
            word_file_path = root_path + '{0}.words.txt'.format(fname)
            tag_file_path = root_path + '{0}.tags.txt'.format(fname)
            dump_list_to_txt(word_file_path,words_list)
            dump_list_to_txt(tag_file_path,tags_list)

def generate_processed_files(file_names,d_flag,u_flag):
    for fname,dflag,uflag in zip(file_names,d_flag,u_flag):
        print("fname: ",fname)
        word_file_path = root_path + '{0}.words.txt'.format(fname)
        tag_file_path = root_path + '{0}.tags.txt'.format(fname)
        output_processed_file_path = '{0}_processed.p'.format(fname)

        # U or validation or test
        if dflag==0:
            print("dflag = = 0")
            fill_lists(word_file_path,tag_file_path,dflag,0,uflag)
            dump_lists_in_pickle(output_processed_file_path)
        else:
            print("dflag = = 1")
            print("u_flag",uflag)
            fill_lists(word_file_path,tag_file_path,dflag,0,uflag)
            fill_lists(rules_center_word_file_path,rules_center_tag_file_path,dflag,1,uflag)
            dump_lists_in_pickle(output_processed_file_path)

generate_word_tag_files() #run once only to create split
generate_processed_files(file_names,d_flag,u_flag)