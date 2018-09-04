
# coding: utf-8

# In[5]:


import os 

def loadupdress():
    idx = 0
    dict ={}
    with open('updress_list.csv') as f:
        for line in f.readlines():
            dict[line.strip()] = idx
            idx+=1
    
    return dict

def loaddowndress():
    idx = 0
    dict ={}
    with open('downdress_list.csv') as f:
        for line in f.readlines():
            dict[line.strip()] = idx
            idx+=1
        
    
    return dict
      
rec_lines =[]
fout= open("attribute_val_pa100k.lst",'w')
'''
with open('attr_label_test.json') as f:
    idx = 0
    updress_dic = loadupdress()
    downdress_dic = loaddowndress()    
    for line in f.readlines():
        import json
        dict = json.loads(line)
        if dict['image_path'][0:4] == 'Mark':
            continue
        img_path = dict['image_path']
        rec_line = '%d\t' % idx
        if 'gender' in dict:
            if 'male' == dict['gender']:
                rec_line +='%f\t' % 0
            else:
                rec_line +='%f\t' % 1
                
        if 'hat' in dict:
            if 'no' == dict['hat']:
                rec_line +='%f\t' % 0
            else:
                rec_line +='%f\t' % 1  
        
        if 'bag' in dict:
            if 'no' == dict['bag']:
                rec_line +='%f\t' % 0
            else:
                rec_line +='%f\t' % 1 
                
        if 'handbag' in dict:
            if 'no' == dict['handbag']:
                rec_line +='%f\t' % 0
            else:
                rec_line +='%f\t' % 1  
                
        if 'backpack' in dict:
            if 'no' == dict['backpack']:
                rec_line +='%f\t' % 0
            else:
                rec_line +='%f\t' % 1  
                
        if 'updress' in dict:
            #print(dict['updress'])
            #print(updress_dic)
            if dict['updress'] in updress_dic:
                rec_line +='%f\t' % updress_dic[dict['updress']]
            else:
                rec_line +='%f\t' % updress_dic['upunknown']
        else:
            rec_line +='%f\t' % updress_dic['upunknown']
            
        if 'downdress' in dict:
            #print(dict['downdress'])
            #print(downdress_dic)
            if dict['downdress'] in downdress_dic:
                rec_line +='%f\t' % downdress_dic[dict['downdress']]
            else:
                rec_line +='%f\t' % downdress_dic['downunknown']
        else:
            rec_line +='%f\t' % downdress_dic['downunknown']
        
        idx+=1
        rec_line += '%s\n' % img_path
        rec_lines.append(rec_line)
'''
        
#add p100k
import scipy.io as sio
anno = sio.loadmat('annotation.mat')
idx2 = 0#idx+1
for i in range(10000):
    rec_line = '%d\t' % idx2
    rec_line +='%f\t' % anno['val_label'][i][0]#female
    rec_line +='%f\t' % anno['val_label'][i][7]#hat
    rec_line +='%f\t' % anno['val_label'][i][10]#PA100K shoulderbag~marketduke bag
    rec_line +='%f\t' % anno['val_label'][i][9]#handbag
    rec_line +='%f\t' % anno['val_label'][i][11]#backpack
    rec_line +='%f\t' % -1 #updress
    rec_line +='%f\t' % -1 #downdress
    img_path = 'PA100K/release_data/release_data/'+ str(anno['val_images_name'][i][0])[2:-2]
    rec_line += '%s\n' % img_path
    idx2+=1
    rec_lines.append(rec_line)
    
        
        
        
import random
random.shuffle(rec_lines)
      
        
for rec_line in rec_lines:        
    fout.write(rec_line)
    #print(rec_line)
fout.close()
print(idx2)

