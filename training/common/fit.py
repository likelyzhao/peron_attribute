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

import mxnet as mx
import logging
import os
import time
import math
from mxboard import SummaryWriter


def get_epoch_size(args, kv):
    return math.ceil(int(args.num_examples / kv.num_workers) / args.batch_size)

def _get_lr_scheduler(args, kv):


    if 'lr_factor' not in args or args.lr_factor >= 1:
        return (args.lr, None)
    epoch_size = get_epoch_size(args, kv)
    begin_epoch = args.load_epoch if args.load_epoch else 0
    if 'pow' in args.lr_step_epochs:
        lr = args.lr
        max_up = args.num_epochs * epoch_size
        pwr = float(re.sub('pow[- ]*', '', args.lr_step_epochs))
        poly_sched = mx.lr_scheduler.PolyScheduler(max_up, lr, pwr)
        return (lr, poly_sched)

    step_epochs = [int(l) for l in args.lr_step_epochs.split(',')]
    lr = args.lr
    for s in step_epochs:
        if begin_epoch >= s:
            lr *= args.lr_factor
    if lr != args.lr:
        logging.info('Adjust learning rate to %e for epoch %d' %(lr, begin_epoch))

    steps = [epoch_size * (x-begin_epoch) for x in step_epochs if x-begin_epoch > 0]
    return (lr, mx.lr_scheduler.MultiFactorScheduler(step=steps, factor=args.lr_factor))

def _load_model(args, rank=0):
    if 'load_epoch' not in args or args.load_epoch is None:
        return (None, None, None)
    assert args.model_prefix is not None
    model_prefix = args.model_prefix
    if rank > 0 and os.path.exists("%s-%d-symbol.json" % (model_prefix, rank)):
        model_prefix += "-%d" % (rank)
    sym, arg_params, aux_params = mx.model.load_checkpoint(
        model_prefix, args.load_epoch)
    logging.info('Loaded model %s_%04d.params', model_prefix, args.load_epoch)
    return (sym, arg_params, aux_params)

def _save_model(args, rank=0):
    if args.model_prefix is None:
        return None
    dst_dir = os.path.dirname(args.model_prefix)
    if not os.path.isdir(dst_dir):
        os.mkdir(dst_dir)
    return mx.callback.do_checkpoint(args.model_prefix if rank == 0 else "%s-%d" % (
        args.model_prefix, rank))

def add_fit_args(parser):
    """
    parser : argparse.ArgumentParser
    return a parser added with args required by fit
    """
    train = parser.add_argument_group('Training', 'model training')
    train.add_argument('--network', type=str,
                       help='the neural network to use')
    train.add_argument('--num-layers', type=int,
                       help='number of layers in the neural network, \
                             required by some networks such as resnet')
    train.add_argument('--gpus', type=str,
                       help='list of gpus to run, e.g. 0 or 0,2,5. empty means using cpu')
    train.add_argument('--kv-store', type=str, default='device',
                       help='key-value store type')
    train.add_argument('--num-epochs', type=int, default=100,
                       help='max num of epochs')
    train.add_argument('--lr', type=float, default=0.1,
                       help='initial learning rate')
    train.add_argument('--lr-factor', type=float, default=0.1,
                       help='the ratio to reduce lr on each step')
    train.add_argument('--lr-step-epochs', type=str,
                       help='the epochs to reduce the lr, e.g. 30,60')
    train.add_argument('--initializer', type=str, default='default',
                       help='the initializer type')
    train.add_argument('--optimizer', type=str, default='sgd',
                       help='the optimizer type')
    train.add_argument('--mom', type=float, default=0.9,
                       help='momentum for sgd')
    train.add_argument('--wd', type=float, default=0.0001,
                       help='weight decay for sgd')
    train.add_argument('--batch-size', type=int, default=128,
                       help='the batch size')
    train.add_argument('--disp-batches', type=int, default=20,
                       help='show progress for every n batches')
    train.add_argument('--model-prefix', type=str,
                       help='model prefix')
    train.add_argument('--save-period', type=int, default=1, help='params saving period')
    train.add_argument('--monitor', dest='monitor', type=int, default=0,
                        help='log network parameters every N iters if larger than 0')
    train.add_argument('--load-epoch', type=int,
                       help='load the model on an epoch using the model-load-prefix')
    train.add_argument('--top-k', type=int, default=0,
                       help='report the top-k accuracy. 0 means no report.')
    train.add_argument('--loss', type=str, default='',
                       help='show the cross-entropy or nll loss. ce strands for cross-entropy, nll-loss stands for likelihood loss')
    train.add_argument('--test-io', type=int, default=0,
                       help='1 means test reading speed without training')
    train.add_argument('--dtype', type=str, default='float32',
                       help='precision: float32 or float16')
    train.add_argument('--gc-type', type=str, default='none',
                       help='type of gradient compression to use, \
                             takes `2bit` or `none` for now')
    train.add_argument('--gc-threshold', type=float, default=0.5,
                       help='threshold for 2bit gradient compression')
    # additional parameters for large batch sgd
    train.add_argument('--macrobatch-size', type=int, default=0,
                       help='distributed effective batch size')
    train.add_argument('--warmup-epochs', type=int, default=5,
                       help='the epochs to ramp-up lr to scaled large-batch value')
    train.add_argument('--warmup-strategy', type=str, default='linear',
                       help='the ramping-up strategy for large batch sgd')
    train.add_argument('--profile-worker-suffix', type=str, default='',
                       help='profile workers actions into this file. During distributed training\
                             filename saved will be rank1_ followed by this suffix')
    train.add_argument('--profile-server-suffix', type=str, default='',
                       help='profile server actions into a file with name like rank1_ followed by this suffix \
                             during distributed training')
    return train
 
class Multi_Acc_Metric(mx.metric.EvalMetric):
    """Calculate accuracies of multi label"""
    def __init__(self, num=None, logger= None,label_names = []):
        self.num = num
        self.logger = logger
        self.label_names= label_names
        super(Multi_Acc_Metric, self).__init__('multi-metric', num=num,label_names= label_names)

    def reset(self):
        """Resets the internal evaluation result to initial state."""
        self.num_inst = 0 if self.num is None else [0] * self.num
        self.sum_metric = 0.0 if self.num is None else [0.0] * self.num

    def update(self, labels, preds):
        # mx.metric.check_label_shapes(labels, preds)
        # classification loss
        #self.logger.info(len(preds))
        for i in range(self.num):
            pred_label = mx.nd.argmax_channel(preds[i])
            pred_label = pred_label.asnumpy().astype('int32')
            label = labels[i].asnumpy().astype('int32')
            #self.logger.info(pred_label)
            #self.logger.info("label")
            #self.logger.info(label)
            mx.metric.check_label_shapes(label, pred_label)

            if self.num is None:
                self.sum_metric += (pred_label.flat == label.flat).sum()
                self.num_inst += len(pred_label.flat)
            else:
                if 0 <= i <= 4:
                    self.sum_metric[i] += (pred_label.flat == label.flat).sum()
                    self.num_inst[i] += len(pred_label.flat)
                if i == 5 or i == 6:
                    self.sum_metric[i] += (pred_label.flat == label.flat).sum()
                    temp = len(pred_label.flat)
                    for item in label:
                        if item == -1:
                            temp-=1
                    self.num_inst[i] += temp


        # verification losses
 #       for i in range(self.cls, len(preds)):
 #           pred = preds[i].asnumpy()
 #           # self.logger.info(pred)
 #           if self.num is None:
 #               self.sum_metric += np.sum(pred)
 #               self.num_inst += len(pred)
 #           else:
 #               self.sum_metric[i] += np.sum(pred)
 #               self.num_inst[i] += len(pred)

    def get(self):
        """Gets the current evaluation result.
        Returns
        -------
        names : list of str
           Name of the metrics.
        values : list of float
           Value of the evaluations.
        """
        #self.logger.info(self.num)
        if self.num is None:
            return super(Multi_Metric, self).get()
        else:
            if len(self.label_names) == self.num:
                loss = [self.sum_metric[i] / ( self.num_inst[i]+ 1e-6) for i in range(self.num)]
                return zip(*(('%s-%s'%(self.name, self.label_names[i]), loss[i]) for i in range(self.num)))
            else:
                #acc = self.sum_metric[0] / self.num_inst[0] if self.num_inst[0] != 0 else float('nan')
                loss = [self.sum_metric[i] / self.num_inst[i] for i in range(self.num)]
                temp =0.0
                for acc in loss:
                    temp +=acc
                acc = temp/len(loss)
                loss = [acc] + loss
            
                return zip(*(('%s-task%d'%(self.name, i), loss[i]) for i in range(1)))

            

    def get_name_value(self):
        """Returns zipped name and value pairs.
        Returns
        -------
        list of tuples
            A (name, value) tuple list.
        """
        if self.num is None:
            return super(Multi_Metric, self).get_name_value()
        name, value = self.get()
        return list(zip(name, value))

class LogMetricsCallback(object):
    """Log metrics periodically in TensorBoard.
    This callback works almost same as `callback.Speedometer`, but write TensorBoard event file
    for visualization. For more usage, please refer https://github.com/dmlc/tensorboard

    Parameters
    ----------
    logging_dir : str
        TensorBoard event file directory.
        After that, use `tensorboard --logdir=path/to/logs` to launch TensorBoard visualization.
    prefix : str
        Prefix for a metric name of `scalar` value.
        You might want to use this param to leverage TensorBoard plot feature,
        where TensorBoard plots different curves in one graph when they have same `name`.
        The follow example shows the usage(how to compare a train and eval metric in a same graph).

    Examples
    --------
    >>> # log train and eval metrics under different directories.
    >>> training_log = 'logs/train'
    >>> evaluation_log = 'logs/eval'
    >>> # in this case, each training and evaluation metric pairs has same name, you can add a prefix
    >>> # to make it separate.
    >>> batch_end_callbacks = [mx.tensorboard.LogMetricsCallback(training_log)]
    >>> eval_end_callbacks = [mx.tensorboard.LogMetricsCallback(evaluation_log)]
    >>> # run
    >>> model.fit(train,
    >>>     ...
    >>>     batch_end_callback = batch_end_callbacks,
    >>>     eval_end_callback  = eval_end_callbacks)
    >>> # Then use `tensorboard --logdir=logs/` to launch TensorBoard visualization.
    """
    def __init__(self, summary_writer, prefix=None):
        self.prefix = prefix
        self.summary_writer =summary_writer
        try:
            from tensorboard import SummaryWriter
            self.summary_writer =summary_writer
        except ImportError:
            logging.error('You can install tensorboard via `pip install tensorboard`.')

    def __call__(self, param):
        """Callback to log training speed and metrics in TensorBoard."""
        if param.eval_metric is None:
            return
        name_value = param.eval_metric.get_name_value()
        print(name_value)
        for name, value in name_value:
            if self.prefix is not None:
                name = '%s-%s' % (self.prefix, name)
            self.summary_writer.add_scalar(tag='accuracy-val',value={name:value},
                global_step=param.epoch)

class Speedometerboradwriter(object):
    """Logs training speed and evaluation metrics periodically.

    Parameters
    ----------
    batch_size: int
        Batch size of data.
    frequent: int
        Specifies how frequently training speed and evaluation metrics
        must be logged. Default behavior is to log once every 50 batches.
    auto_reset : bool
        Reset the evaluation metrics after each log.

    """
    def __init__(self, batch_size, summary_writer, frequent=50, auto_reset=True):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.auto_reset = auto_reset
        self.summary_writer = summary_writer

    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                # #11504
                try:
                    speed = self.frequent * self.batch_size / (time.time() - self.tic)
                except ZeroDivisionError:
                    speed = float('inf')
                if param.eval_metric is not None:
                    name_value = param.eval_metric.get_name_value()
                    if self.auto_reset:
                        param.eval_metric.reset()
                        global_step = (param.epoch * param.nbatch + count)/self.batch_size
                        self.summary_writer.add_scalar(tag='speed', 
                            value={'speed':speed}, global_step=global_step)

                        for name in name_value:
                            self.summary_writer.add_scalar(tag='accuracy', 
                                value={name[0]:name[1]}, global_step=global_step)
                    else:
                        msg = 'Epoch[%d] Batch [0-%d]\tSpeed: %.2f samples/sec'
                        msg += '\t%s=%f'*len(name_value)
                        #logging.info(msg, param.epoch, count, speed, *sum(name_value, ()))
                else:
                    pass
                    #logging.info("Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec",
                                 #param.epoch, count, speed)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()
   
def fit(args, network, data_loader, **kwargs):
    """
    train a model
    args : argparse returns
    network : the symbol definition of the nerual network
    data_loader : function that returns the train and val data iterators
    """
    # mxborad  
    sw_train = SummaryWriter(logdir='./logs/train', flush_secs=20)
    sw_val = SummaryWriter(logdir='./logs/val', flush_secs=20)
    sw_image= SummaryWriter(logdir='./logs/image', flush_secs=20)
    sw_symbol= SummaryWriter(logdir='./logs/symbol', flush_secs=20)
    sw_symbol.add_graph(network)
    args.summary_writer_image = None

    # kvstore
    kv = mx.kvstore.create(args.kv_store)
    if args.gc_type != 'none':
        kv.set_gradient_compression({'type': args.gc_type,
                                     'threshold': args.gc_threshold})

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=head)
    logging.info('start with arguments %s', args)

    #epoch_size 
    epoch_size = get_epoch_size(args, kv)

    # data iterators

    (train, val) = data_loader(args, kv)
    if 'dist' in args.kv_store and not 'async' in args.kv_store:
        logging.info('Resizing training data to %d batches per machine', epoch_size)
        # resize train iter to ensure each machine has same number of batches per epoch
        # if not, dist_sync can hang at the end with one machine waiting for other machines
        train = mx.io.ResizeIter(train, epoch_size)

    if args.test_io:
        tic = time.time()
        for i, batch in enumerate(train):
            for j in batch.data:
                j.wait_to_read()
            if (i+1) % args.disp_batches == 0:
                logging.info('Batch [%d]\tSpeed: %.2f samples/sec' % (
                    i, args.disp_batches*args.batch_size/(time.time()-tic)))
                tic = time.time()

        return


    print('next_sample',train.next())
    # load model
    if 'arg_params' in kwargs and 'aux_params' in kwargs:
        arg_params = kwargs['arg_params']
        aux_params = kwargs['aux_params']
    else:
        sym, arg_params, aux_params = _load_model(args, kv.rank)
        if sym is not None:
            assert sym.tojson() == network.tojson()

    # save model
    checkpoint = _save_model(args, kv.rank)

    # devices for training
    devs = mx.cpu() if args.gpus is None or args.gpus is '' else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    # learning rate
    lr, lr_scheduler = _get_lr_scheduler(args, kv)

    # create model
    model = mx.mod.Module(
        context       = devs,
        symbol        = network,
        label_names   = {'gender_label','hat_label','bag_label','handbag_label','backpack_label','updress_label','downdress_label'}
    )
    model_mix = mx.mod.Module(
        context       = devs,
        symbol        = network,
        label_names   = {'gender_label','hat_label','bag_label','handbag_label','backpack_label','updress_label','downdress_label',
                         'gender_mix_label','hat_mix_label','bag_mix_label','handbag_mix_label','backpack_mix_label','updress_mix_label','downdress_mix_label'}
    )

    lr_scheduler  = lr_scheduler
    optimizer_params = {
        'learning_rate': lr,
        'wd' : args.wd,
        'lr_scheduler': lr_scheduler,
        'multi_precision': True}

    # Only a limited number of optimizers have 'momentum' property
    has_momentum = {'sgd', 'dcasgd', 'nag', 'signum', 'lbsgd'}
    if args.optimizer in has_momentum:
        optimizer_params['momentum'] = args.mom

    monitor = mx.mon.Monitor(args.monitor, pattern=".*") if args.monitor > 0 else None

     # A limited number of optimizers have a warmup period
    has_warmup = {'lbsgd', 'lbnag'}
    if args.optimizer in has_warmup:
        nworkers = kv.num_workers
        if epoch_size < 1:
            epoch_size = 1
        macrobatch_size = args.macrobatch_size
        if macrobatch_size < args.batch_size * nworkers:
            macrobatch_size = args.batch_size * nworkers
        #batch_scale = round(float(macrobatch_size) / args.batch_size / nworkers +0.4999)
        batch_scale = math.ceil(
            float(macrobatch_size) / args.batch_size / nworkers)
        optimizer_params['updates_per_epoch'] = epoch_size
        optimizer_params['begin_epoch'] = args.load_epoch if args.load_epoch else 0
        optimizer_params['batch_scale'] = batch_scale
        optimizer_params['warmup_strategy'] = args.warmup_strategy
        optimizer_params['warmup_epochs'] = args.warmup_epochs
        optimizer_params['num_epochs'] = args.num_epochs


    if args.initializer == 'default':
        if args.network == 'alexnet':
            # AlexNet will not converge using Xavier
            initializer = mx.init.Normal()
            # VGG will not trend to converge using Xavier-Gaussian
        elif args.network and 'vgg' in args.network:
            initializer = mx.init.Xavier()
        else:
            initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2)
    # initializer   = mx.init.Xavier(factor_type="in", magnitude=2.34),
    elif args.initializer == 'xavier':
        initializer = mx.init.Xavier()
    elif args.initializer == 'msra':
        initializer = mx.init.MSRAPrelu()
    elif args.initializer == 'orthogonal':
        initializer = mx.init.Orthogonal()
    elif args.initializer == 'normal':
        initializer = mx.init.Normal()
    elif args.initializer == 'uniform':
        initializer = mx.init.Uniform()
    elif args.initializer == 'one':
        initializer = mx.init.One()
    elif args.initializer == 'zero':
        initializer = mx.init.Zero()


    # evaluation metrices
    eval_metrics = ['accuracy']
    if args.top_k > 0:
        eval_metrics.append(mx.metric.create('top_k_accuracy', top_k=args.top_k))
        
    eval_metrics = Multi_Acc_Metric(num=7,label_names= ['gender_label','hat_label','bag_label','handbag_label','backpack_label','updress_label','downdress_label'])

    supported_loss = ['ce', 'nll_loss']
    if len(args.loss) > 0:
        # ce or nll loss is only applicable to softmax output
        loss_type_list = args.loss.split(',')
        if 'gender_label_output' in network.list_outputs():
            for loss_type in loss_type_list:
                loss_type = loss_type.strip()
                if loss_type == 'nll':
                    loss_type = 'nll_loss'
                if loss_type not in supported_loss:
                    logging.warning(loss_type + ' is not an valid loss type, only cross-entropy or ' \
                                    'negative likelihood loss is supported!')
                else:
                    eval_metrics.append(mx.metric.create(loss_type))
        else:
            logging.warning("The output is not softmax_output, loss argument will be skipped!")


    # callbacks that run after each batch
    batch_end_callbacks = [mx.callback.Speedometer(args.batch_size, args.disp_batches,False)]
    batch_end_callback_with_sw = [Speedometerboradwriter(args.batch_size,sw_train,args.disp_batches)]
    eval_end_callback_with_sw = [LogMetricsCallback(sw_val)]
    if 'batch_end_callback' in kwargs:
        cbs = kwargs['batch_end_callback']
        batch_end_callbacks += cbs if isinstance(cbs, list) else [cbs]
    batch_end_callbacks +=batch_end_callback_with_sw
    print(batch_end_callbacks)

    # run
    model_mix.fit(train,
              begin_epoch        = args.load_epoch if args.load_epoch else 0,
              num_epoch          = args.num_epochs,
              eval_data          = val,
              eval_metric        = eval_metrics,
              kvstore            = kv,
              optimizer          = args.optimizer,
              optimizer_params   = optimizer_params,
              initializer        = initializer,
              arg_params         = arg_params,
              aux_params         = aux_params,
              batch_end_callback = batch_end_callbacks,
              eval_end_callback  = eval_end_callback_with_sw,
              epoch_end_callback = checkpoint,
              allow_missing      = True,
              monitor            = monitor)
