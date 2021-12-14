'''Implements a generic training loop.
'''

import torch
import utils
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil

def train_sdfc2(model, train_dataloader, epochs, lr, steps_til_summary,
          epochs_til_checkpoint, model_dir, loss_shape, loss_color,
          clip_grad=False, loss_schedules=None):

    # separate optimizers
    shape_params=list(model.net_sdf.parameters())
    shape_params.append(model.embd.weight)
    color_params=list(model.net_rgb.parameters())
    color_params.append(model.embdc.weight)
    optim_shape = torch.optim.Adam(shape_params,lr=lr)
    optim_color = torch.optim.Adam(color_params,lr=lr)

    #optim = torch.optim.Adam(lr=lr, params=model.parameters())

    if os.path.exists(model_dir):
        val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        if val == 'y':
            shutil.rmtree(model_dir)
    os.makedirs(model_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}
                 
                model_in, sdf, rgb = model(model_input)
                loss_s = loss_shape({'model_in': model_in['coords'],
                                  'model_out': sdf},
                                 gt)
                loss_c = loss_color({'model_in': model_in['coords'],'model_out_rgb':rgb},gt)

                shape_loss=0
                color_loss=0
                for loss_name, loss in loss_s.items():
                    single_loss = loss.mean()
                    shape_loss += single_loss

                color_loss = loss_c['rgb'].mean()
                
                torch.autograd.set_detect_anomaly(True)

                optim_shape.zero_grad()
                shape_loss.backward(retain_graph=True)
                optim_shape.step()

                optim_color.zero_grad()
                color_loss.backward()
                optim_color.step()
                #import pdb; pdb.set_trace()                
                #color_loss = loss_c['rgb'].mean()
                #color_loss.backward()
                #optim_color.step()

                train_loss = color_loss.item()+shape_loss.item()
                train_losses.append(train_loss.item())

                if not total_steps % steps_til_summary:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))

                #if clip_grad:
                #    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                
                pbar.update(1)
                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                total_steps += 1

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))


def train_sdfc(model, train_dataloader, epochs, lr, steps_til_summary,
          epochs_til_checkpoint, model_dir, loss_fn,
          clip_grad=False, loss_schedules=None):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    if os.path.exists(model_dir):
        val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        if val == 'y':
            shutil.rmtree(model_dir)
    os.makedirs(model_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}
                 
                model_in, sdf, rgb = model(model_input)
                losses = loss_fn({'model_in': model_in['coords'],
                                  'model_out': sdf,
                                  'model_out_rgb': rgb},
                                 gt)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    train_loss += single_loss

                train_losses.append(train_loss.item())

                if not total_steps % steps_til_summary:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))

                optim.zero_grad()
                train_loss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                optim.step()
                pbar.update(1)
                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                total_steps += 1

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))



def train_sdf(model, train_dataloader, epochs, lr, steps_til_summary,
          epochs_til_checkpoint, model_dir, loss_fn,
          clip_grad=False, loss_schedules=None):

    optim = torch.optim.Adam(lr=lr, params=model.parameters())

    if os.path.exists(model_dir):
        val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        if val == 'y':
            shutil.rmtree(model_dir)
    os.makedirs(model_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    utils.cond_mkdir(checkpoints_dir)

    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        train_losses = []
        for epoch in range(epochs):
            if not epoch % epochs_til_checkpoint and epoch:
                torch.save(model.state_dict(),
                           os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % epoch))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % epoch),
                           np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()

            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}
                 

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()
                    train_loss += single_loss

                train_losses.append(train_loss.item())

                if not total_steps % steps_til_summary:
                    torch.save(model.state_dict(),
                               os.path.join(checkpoints_dir, 'model_current.pth'))

                optim.zero_grad()
                train_loss.backward()
                if clip_grad:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
                optim.step()
                pbar.update(1)
                if not total_steps % steps_til_summary:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))
                total_steps += 1

        torch.save(model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))
