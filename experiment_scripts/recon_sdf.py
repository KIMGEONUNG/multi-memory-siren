import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import sdf_meshing
import configargparse

from tqdm import tqdm
import numpy as np
import dataio, loss_functions 
from torch.utils.data import DataLoader
from torch.autograd import Variable


def parse():
    p = configargparse.ArgumentParser()
    p.add('-c', '--config_filepath', required=False, is_config_file=True, 
            help='Path to config file.')

    p.add_argument('--logging_root', type=str, default='./logs', 
                    help='root for logging')
    p.add_argument('--experiment_name', type=str, required=True,
                   help='Name of subdirectory in logging_root \
                   where summaries and checkpoints will be saved.')

    # General training options
    p.add_argument('--checkpoint_path', default=None, 
            help='Checkpoint to trained model.')
    p.add_argument('--resolution', type=int, default=1600)
    p.add_argument('--lr', type=float, default=1e-4, 
            help='learning rate. default=5e-5')

    ################### Check this option ##################
    p.add_argument('--batch_size', type=int, default=1400)
    p.add_argument('--path_target', type=str, required=True,
                   help='reconstruction target')
    p.add_argument('--dim_embd', type=int, default=7)
    p.add_argument('--num_class', type=int, default=20)
    p.add_argument('--dim_hidden', type=int, default=256)
    p.add_argument('--num_layer', type=int, default=3)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_iter', type=int, default=2000)
    ########################################################
    opt = p.parse_args()
    return opt


def main(opt):
    # Seed
    if opt.seed >= 0:
        utils.set_seed(opt.seed)

    # IO setting
    root_path = os.path.join(opt.logging_root, opt.experiment_name)
    utils.cond_mkdir(root_path)

    # Define the model.
    model = modules.SDFDecoder(opt.num_class,
                               opt.dim_embd,
                               opt.dim_hidden,
                               opt.num_layer)
    model.load_state_dict(torch.load(opt.checkpoint_path))
    model.eval()
    model.cuda()

    # Sample init latent vector
    code = torch.randn(opt.dim_embd)
    code = code.cuda()

    sdf_dataset = dataio.PointCloudSingle(opt.path_target, on_surface_points=opt.batch_size)
    dataloader = DataLoader(sdf_dataset, shuffle=True,
                            batch_size=1, pin_memory=True, num_workers=0)

    loss_fn = loss_functions.sdf
    root_path = os.path.join(opt.logging_root, opt.experiment_name)

    code = optimize(model=model,
             code=code,
             train_dataloader=dataloader,
             epochs=opt.num_iter,
             lr=opt.lr,
             model_dir=root_path,
             loss_fn=loss_fn,
             clip_grad=True)
    print('Optimization finished')

    sdf_meshing.create_mesh_with_code(model,
                                      os.path.join(root_path, 'test'),
                                      code=code,
                                      N=opt.resolution)
    print('Reconstruction finished')


def optimize(model, code, train_dataloader, epochs, lr, model_dir, loss_fn,
          clip_grad=False):

    model.eval() 

    code = Variable(code, requires_grad=True)
    optim = torch.optim.Adam(lr=lr, params=[code])
    optim.zero_grad()

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            for step, (model_input, gt) in enumerate(train_dataloader):

            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}
                coords = model_input['coords']

                model_output = model.forward_with_code(coords, code)
                losses = loss_fn(model_output, gt)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    train_loss += single_loss
                optim.zero_grad()
                train_loss.backward()

                optim.step()
                pbar.update(1)
                pbar.set_description('loss: %8.4f' % train_loss.item())
                total_steps += 1
    return code


if __name__ == '__main__':
    opt = parse()
    main(opt)
