# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import os
import argparse
import logging
logging.basicConfig(level=logging.DEBUG)
from common import find_mxnet
from common import data, fit, modelzoo
import mxnet as mx

def get_fine_tune_model(symbol, arg_params, num_classes, layer_name):
    """
    symbol: the pre-trained network symbol
    arg_params: the argument parameters of the pre-trained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    
    #downdress_label = mx.sym.Variable('downdress_label')
    
    all_layers = symbol.get_internals()
    net = all_layers[layer_name+'_output']
    num_hidden =  1024
    
    net_gender_split = net
    net_gender = mx.symbol.Dropout(net_gender_split, p = 0.5)
    net_gender_hidden = mx.symbol.FullyConnected(data=net_gender_split, num_hidden=num_hidden, name='fc-gender-hidden')
    net_gender_fc = mx.symbol.FullyConnected(data=net_gender_hidden, num_hidden=2, name='fc-gender')
    #net_gender_fc = mx.symbol.BlockGrad(net_gender_fc)
    net_gender = mx.symbol.SoftmaxOutput(data=net_gender_fc, name='gender',grad_scale=0.1)
    
    net_hat_split = net
    net_hat = mx.symbol.Dropout(net_hat_split, p = 0.5)
    net_hat_hidden = mx.symbol.FullyConnected(data=net_hat_split, num_hidden=num_hidden, name='fc-hat-hidden')
    net_hat_fc = mx.symbol.FullyConnected(data=net_hat_hidden, num_hidden=2, name='fc-hat')
    #net_hat_fc = mx.symbol.BlockGrad(net_hat_fc)
    net_hat = mx.symbol.SoftmaxOutput(data=net_hat_fc, name='hat',grad_scale=0.1)
    
    net_bag_split = net
    net_bag = mx.symbol.Dropout(net_bag_split, p = 0.5)
    net_bag_hidden = mx.symbol.FullyConnected(data=net_bag_split, num_hidden=num_hidden, name='fc-bag-hidden')
    net_bag_fc = mx.symbol.FullyConnected(data=net_bag_hidden, num_hidden=2, name='fc-bag')
    #net_bag_fc = mx.symbol.BlockGrad(net_bag_fc)
    net_bag = mx.symbol.SoftmaxOutput(data=net_bag_fc, name='bag',grad_scale=0.1)
    
    net_handbag_split = net
    net_handbag = mx.symbol.Dropout(net_handbag_split, p = 0.5)
    net_handbag_hidden = mx.symbol.FullyConnected(data=net_handbag_split, num_hidden=num_hidden, name='fc-handbag-hidden')
    net_handbag_fc = mx.symbol.FullyConnected(data=net_handbag_hidden, num_hidden=2, name='fc-handbag')
    #net_handbag_fc = mx.symbol.BlockGrad(net_handbag_fc)
    net_handbag = mx.symbol.SoftmaxOutput(data=net_handbag_fc, name='handbag',grad_scale=0.1)
    
    net_backpack_split = net
    net_backpack = mx.symbol.Dropout(net_backpack_split, p = 0.5)
    net_backpack_hidden = mx.symbol.FullyConnected(data=net_backpack_split, num_hidden=num_hidden, name='fc-backpack-hidden')
    net_backpack_fc= mx.symbol.FullyConnected(data=net_backpack_hidden, num_hidden=2, name='fc-backpack')
    #net_backpack_fc = mx.symbol.BlockGrad(net_backpack_fc)
    net_backpack = mx.symbol.SoftmaxOutput(data=net_backpack_fc, name='backpack',grad_scale=0.1)
    
    net_updress_split = net
    net_updress = mx.symbol.Dropout(net_updress_split, p = 0.5)
    net_updress_hidden = mx.symbol.FullyConnected(data=net_updress_split, num_hidden=num_hidden, name='fc-updress-hidden')
    net_updress_fc = mx.symbol.FullyConnected(data=net_updress_hidden, num_hidden=7, name='fc-updress')
    #net_updress_fc = mx.symbol.BlockGrad(net_updress_fc)
    net_updress = mx.symbol.SoftmaxOutput(data=net_updress_fc, name='updress', ignore_label=-1,use_ignore=True)

    net_downdress_split = net
    net_downdress = mx.symbol.Dropout(net_downdress_split, p = 0.5)
    net_downdress_hidden = mx.symbol.FullyConnected(data=net_downdress_split, num_hidden=num_hidden, name='fc-downdress-hidden')
    net_downdress_fc = mx.symbol.FullyConnected(data=net_downdress_hidden, num_hidden=10, name='fc-downdress')
    #net_downdress_fc = mx.symbol.BlockGrad(net_downdress_fc)
    net_downdress = mx.symbol.SoftmaxOutput(data=net_downdress_fc, name='downdress',ignore_label= -1,use_ignore=True)
    
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc' not in k})
    
    
    #net = mx.symbol.Group([mx.symbol.BlockGrad(net_gender), mx.symbol.BlockGrad(net_hat), mx.symbol.BlockGrad(net_bag),
    #                       mx.symbol.BlockGrad(net_handbag), mx.symbol.BlockGrad(net_backpack)
    #                       ,mx.symbol.BlockGrad(net_updress), mx.symbol.BlockGrad(net_downdress)])
    net = mx.symbol.Group([net_gender, net_hat, net_bag, net_handbag, net_backpack,net_updress, net_downdress])
    
    return (net, new_args)


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="fine-tune a dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    train = fit.add_fit_args(parser)
    data.add_data_args(parser)
    aug = data.add_data_aug_args(parser)
    parser.add_argument('--pretrained-model', type=str,
                        help='the pre-trained model')
    parser.add_argument('--layer-before-fullc', type=str, default='flatten0',
                        help='the name of the layer before the last fullc layer')
    # use less augmentations for fine-tune
    data.set_data_aug_level(parser, 1)
    # use a small learning rate and less regularizations
    parser.set_defaults(image_shape='3,224,224', num_epochs=30,
                        lr=.01, lr_step_epochs='20', wd=0, mom=0)

    args = parser.parse_args()

    # load pretrained model
    dir_path = os.path.dirname(os.path.realpath(__file__))
    (prefix, epoch) = modelzoo.download_model(
        args.pretrained_model, os.path.join(dir_path, 'model'))
    if args.load_epoch is not None:
        (prefix, epoch) = (args.model_prefix, args.load_epoch)
    logging.info(prefix)
    logging.info(epoch)
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)

    # remove the last fullc layer
    (new_sym, new_args) = get_fine_tune_model(
        sym, arg_params, args.num_classes, args.layer_before_fullc)
    
    arg_shape,out_shape,aux_shape = new_sym.infer_shape(data=(16,3,112,112))
    print("out-shape")
    print(out_shape)

    print("arg_shape")
    print(arg_shape)
    
    
    # train
    fit.fit(args        = args,
            network     = new_sym,
            data_loader = data.get_rec_iter_mutil,
            arg_params  = new_args,
            aux_params  = aux_params)
