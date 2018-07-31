import mxnet as mx
import numpy as np
import cv2
import os
import time

json_file = "model/personattr-symbol.json"
param_file = "model/personattr-0049.params"
P100K_data_path = 'release_data'
P100K_list_file = 'list.txt'
input_shape  = [1,3,112,112]

r_mean = 123.68
g_mean = 116.779
b_mean = 103.939

net, arg_params,aux_params = mx.model.load_checkpoint('model/personattr',49)

label_names = ['gender_label','hat_label','bag_label','handbag_label','backpack_label','updress_label','downdress_label']

label_dicts  = {'gender_label':['male','female'],
          'hat_label':['no_hat','hat'],
          'bag_label':['no_bag','bag'],
          'handbag_label':['no_handbag','handbag'],
          'backpack_label':['no_backpack','backpack'],
          'updress_label':['upunknown','upblack','upblue','upgreen','upgray','uppurple','upred','upyellow'],
          'downdress_label':['downunknown','downblack','downblue','downbrown','downgray','downgreen','downpink','downpurple',
                                'downwhite','downred','downyellow']
        
         }

mod = mx.mod.Module(net,label_names=('gender_label','hat_label','bag_label','handbag_label','backpack_label','updress_label','downdress_label'),
                   context= mx.gpu(0))

data_input = [mx.io.DataDesc(str('data'), input_shape,layout='NCHW')]
print(data_input)
#mod.infer_shape(data_input)
mod.bind(data_input,for_training=False)
mod.set_params(arg_params,aux_params)
fout = open("save.json",'w')

with open(P100K_list_file) as f:
    for line in f.readlines():
        time_start=time.time()

        img_path  = os.path.join(P100K_data_path,line.strip())
        im  = cv2.imread(img_path)
        im  = cv2.resize(im,(128,128))
        im  = im[7:119,7:119,:]
        
        input_buffer = mx.nd.empty(input_shape)
        #im[:,:,2] = im[:,:,2]  = r_mean
        input_buffer[0,0,:,:] = mx.nd.array(im[:,:,2] - r_mean)
        input_buffer[0,1,:,:] = mx.nd.array(im[:,:,1] - g_mean)
        input_buffer[0,2,:,:] = mx.nd.array(im[:,:,0] - b_mean)
            
        data = mx.io.DataBatch([input_buffer])
        mod.forward(data)
        single_dict ={}
        single_dict['url'] = img_path
        data = []
        for idx,output in  enumerate(mod.get_outputs()):
            # print(label_names[idx])
            temp = output.asnumpy()
            max_idx = np.argmax(temp)
            label_name = label_dicts[label_names[idx]][max_idx]
            score = temp[0,max_idx]
            if len(temp[0,:]) >2:
                score *= len(temp[0,:]) 
                     
            #print(label_names[idx] + ' : ' + label_dicts[label_names[idx]][max_idx])
            data.append({"class":label_dicts[label_names[idx]][max_idx],"score": str(score)})
            #print(temp)
            #print(score)
            
        single_dict['data'] = data
        print(single_dict)
        import json
        fout.write(json.dumps(single_dict) + '\n')
        time_end=time.time()
        #print('totally cost',time_end-time_start)
        
        
            
    
        
        
    

