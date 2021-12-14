import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import modules, utils
import sdf_meshing
import configargparse

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

################### Check this option ##################
p.add_argument('--dim_embd', type=int, default=7)
p.add_argument('--dim_embdc', type=int, default=None)
p.add_argument('--c2_conditioned',type=bool, default=True)
p.add_argument('--num_class', type=int, default=20)
p.add_argument('--dim_hidden', type=int, default=256)
p.add_argument('--num_layer', type=int, default=3)
p.add_argument('--dropout', type=float, default=None)
########################################################

opt = p.parse_args()

# Define the model.
if opt.dim_embdc==None:
    model = modules.SDFCDecoder(opt.num_class,
                           opt.dim_embd,
                           opt.dim_hidden,
                           opt.num_layer,
                           opt.dropout)
else:
    model = modules.SDFC2Decoder(opt.num_class,
                           opt.dim_embd,
                           opt.dim_embdc,
                           opt.c2_conditioned,
                           opt.dim_hidden,
                           opt.num_layer,
                           opt.dropout)
model.load_state_dict(torch.load(opt.checkpoint_path))
model.cuda()

root_path = os.path.join(opt.logging_root, opt.experiment_name)
utils.cond_mkdir(root_path)

codes = list(range(opt.num_class))
sdf_meshing.create_mesh_rgb_multi(model,
                                  os.path.join(root_path, 'test'),
                                  codes=codes,
                                  N=opt.resolution)
