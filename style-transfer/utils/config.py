from pprint import pprint

class Config(object):
    # General Args
    use_gpu = True
    model_path = None # pretrain model path (for resume training or test)
    
    # Train Args
    image_size = 256 # image crop_size for training
    batch_size = 8  
    data_root = 'data/' # dataset root：$data_root/coco/a.jpg
    num_workers = 4 # dataloader num of workers
    
    lr = 1e-3
    epoches = 2 # total epoch to train
    content_weight = 1e5 # weight of content_loss  
    style_weight = 1e10 # weight of style_loss

    style_path= 'style.jpg' # style image path
    env = 'style-transfer' # visdom env
    plot_every = 10 # visualize in visdom for every 10 batch

    debug_file = '/tmp/debugnn' # touch $debug_fie to interrupt and enter ipdb 

    # Test Args
    content_path = 'input.png' # input file to do style transfer [for test]
    result_path = 'output.png' # style transfer result [for test]

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

opt = Config()

