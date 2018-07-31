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
    all_layers = symbol.get_internals()
    net = all_layers[layer_name+'_output']
    num_hidden = 64 
    
    net_gender = mx.symbol.Dropout(net, p = 0.5)
    net_gender = mx.symbol.FullyConnected(data=net_gender, num_hidden=num_hidden, name='fc-gender-hidden')
    net_gender = mx.symbol.FullyConnected(data=net_gender, num_hidden=2, name='fc-gender')
    net_gender = mx.symbol.SoftmaxOutput(data=net_gender, name='gender')
    
    net_hat = mx.symbol.Dropout(net, p = 0.5)
    net_hat = mx.symbol.FullyConnected(data=net_hat, num_hidden=num_hidden, name='fc-hat-hidden')
    net_hat = mx.symbol.FullyConnected(data=net_hat, num_hidden=2, name='fc-hat')
    net_hat = mx.symbol.SoftmaxOutput(data=net_hat, name='hat')
    
    net_bag = mx.symbol.Dropout(net, p = 0.5)
    net_bag = mx.symbol.FullyConnected(data=net_bag, num_hidden=num_hidden, name='fc-bag-hidden')
    net_bag = mx.symbol.FullyConnected(data=net_bag, num_hidden=2, name='fc-bag')
    net_bag = mx.symbol.SoftmaxOutput(data=net_bag, name='bag')
    
    net_handbag = mx.symbol.Dropout(net, p = 0.5)
    net_handbag = mx.symbol.FullyConnected(data=net_handbag, num_hidden=num_hidden, name='fc-handbag-hidden')
    net_handbag = mx.symbol.FullyConnected(data=net_handbag, num_hidden=2, name='fc-handbag')
    net_handbag = mx.symbol.SoftmaxOutput(data=net_handbag, name='handbag')
    
    net_backpack = mx.symbol.Dropout(net, p = 0.5)
    net_backpack = mx.symbol.FullyConnected(data=net_backpack, num_hidden=num_hidden, name='fc-backpack-hidden')
    net_backpack= mx.symbol.FullyConnected(data=net_backpack, num_hidden=2, name='fc-backpack')
    net_backpack = mx.symbol.SoftmaxOutput(data=net_backpack, name='backpack')
    

    net_updress = mx.symbol.Dropout(net, p = 0.5)
    net_updress = mx.symbol.FullyConnected(data=net_updress, num_hidden=num_hidden, name='fc-updress-hidden')
    net_updress = mx.symbol.FullyConnected(data=net_updress, num_hidden=8, name='fc-updress')
    net_updress = mx.symbol.SoftmaxOutput(data=net_updress, name='updress')

    net_downdress = mx.symbol.Dropout(net, p = 0.5)
    net_downdress = mx.symbol.FullyConnected(data=net_downdress, num_hidden=num_hidden, name='fc-downdress-hidden')
    net_downdress = mx.symbol.FullyConnected(data=net_downdress, num_hidden=11, name='fc-downdress')
    net_downdress = mx.symbol.SoftmaxOutput(data=net_downdress, name='downdress')
    
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc' not in k})
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

    # train
    fit.fit(args        = args,
            network     = new_sym,
            data_loader = data.get_rec_iter_mutil,
            arg_params  = new_args,
            aux_params  = aux_params)
