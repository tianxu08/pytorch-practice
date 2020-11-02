# coding:utf8

import torch as t
import torchvision as tv
import torchnet as tnt
import tqdm
import os
import ipdb

from utils.config import opt
from utils.vis_tool import Visualizer
from torch.utils import data
from models.transformer_net import TransformerNet
from models.WrappedVGG import Vgg16

from torch.nn import functional as F
import utils.utils as utils

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def train(**kwargs):
    opt._parse(kwargs)
    
    device=t.device('cuda') if opt.use_gpu else t.device('cpu')
    vis = Visualizer(opt.env)

    # Data loading
    transfroms = tv.transforms.Compose([
        tv.transforms.Resize(opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x * 255)
    ])
    dataset = tv.datasets.ImageFolder(opt.data_root, transfroms)
    dataloader = data.DataLoader(dataset, opt.batch_size)

    # style transformer network
    transformer = TransformerNet()
    if opt.model_path:
        transformer.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))
    transformer.to(device)

    # Vgg16 for Perceptual Loss
    vgg = Vgg16().eval()
    vgg.to(device)
    for param in vgg.parameters():
        param.requires_grad = False

    # Optimizer: use Adam
    optimizer = t.optim.Adam(transformer.parameters(), opt.lr)

    # Get style image
    style = utils.get_style_data(opt.style_path)
    vis.img('style', (style.data[0] * 0.225 + 0.45).clamp(min=0, max=1))
    style = style.to(device)

    # print("style.shape: ", style.shape)

    # gram matrix for style image
    with t.no_grad():
        features_style = vgg(style)
        gram_style = [utils.gram_matrix(y) for y in features_style]

    # Loss meter
    style_meter = tnt.meter.AverageValueMeter()
    content_meter = tnt.meter.AverageValueMeter()

    for epoch in range(opt.epoches):
        content_meter.reset()
        style_meter.reset()

        for ii, (x, _) in tqdm.tqdm(enumerate(dataloader)):

            # Train
            optimizer.zero_grad()
            x = x.to(device)
            y = transformer(x)
            y = utils.normalize_batch(y)
            x = utils.normalize_batch(x)
            features_y = vgg(y)
            features_x = vgg(x)

            # content loss
            content_loss = opt.content_weight * F.mse_loss(features_y.relu2_2, features_x.relu2_2)

            # style loss
            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gram_y = utils.gram_matrix(ft_y)
                style_loss += F.mse_loss(gram_y, gm_s.expand_as(gram_y))
            style_loss *= opt.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            # Loss smooth for visualization
            content_meter.add(content_loss.item())
            style_meter.add(style_loss.item())

            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # visualization
                vis.plot('content_loss', content_meter.value()[0])
                vis.plot('style_loss', style_meter.value()[0])
                # denorm input/output, since we have applied (utils.normalize_batch)
                vis.img('output', (y.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))
                vis.img('input', (x.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))

        # save checkpoint
        vis.save([opt.env])
        t.save(transformer.state_dict(), 'checkpoints/%s_style.pth' % epoch)

@t.no_grad()
def test(**kwargs):
    """
    perform style transfer
    """
    opt = Config()

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)
    device=t.device('cuda') if opt.use_gpu else t.device('cpu')
    
    # preprocess input image 
    content_image = tv.datasets.folder.default_loader(opt.content_path)
    content_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0).to(device).detach()

    # model setup
    style_model = TransformerNet().eval()
    style_model.load_state_dict(t.load(opt.model_path, map_location=lambda _s, _: _s))
    style_model.to(device)

    # style transfer and save output
    output = style_model(content_image)
    output_data = output.cpu().data[0]
    tv.utils.save_image(((output_data / 255)).clamp(min=0, max=1), opt.result_path)


if __name__ == '__main__':
    import fire
    fire.Fire()
