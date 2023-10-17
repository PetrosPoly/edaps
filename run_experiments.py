# ------------------------------------------------------------------------------------
# Copyright (c) 2022-2023 ETH Zurich, Suman Saha, Lukas Hoyer. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------------------

import argparse
import json
import os
import subprocess
import uuid
from datetime import datetime
import torch
from mmcv import Config, get_git_hash
from tools import train_mmdet
from tools import train
from tools import test_mmdet
from tools import test
from tools.panoptic_deeplab.utils import create_panop_eval_folders
from experiments import generate_experiment_cfgs
from experiments_bottomup import generate_experiment_cfgs as generate_experiment_cfgs_bottomup
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.autograd.set_detect_anomaly(True)


def run_command(command):
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    for line in iter(p.stdout.readline, b''):
        print(line.decode('utf-8'), end='')


def rsync(src, dst):
    rsync_cmd = f'rsync -a {src} {dst}'
    print(rsync_cmd)
    run_command(rsync_cmd)

""" 
def run_command(command):
    # 1st line takes one argument, command, which is expected to be a string representing a command that you want to run in the system shell.
    # 2nd line creates a new process (Popen) to run the specified command. command is the command to be executed. It's passed as a string.
    # stdout=subprocess.PIPE redirects the standard output of the command to a pipe so that it can be captured.
    # stderr=subprocess.STDOUT redirects the standard error (if any) to the same pipe as the standard output.
    # Line starts a loop that reads the output of the executed command line by line.
    # This line prints the constructed rsync command before it is executed.
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True)
    for line in iter(p.stdout.readline, b''):
        print(line.decode('utf-8'), end='')

def rsync(src, dst):
    # This line calls the run_command function with the constructed rsync command as an argument, which then executes the rsync command and captures its output.
    rsync_cmd = f'rsync -a {src} {dst}'
    print(rsync_cmd)
    run_command(rsync_cmd)
"""

if __name__ == '__main__':

    config_file = None
    expId = None
    #machine_name = None
    machine_name = 'local'
    JOB_DIR = 'jobs'
    WORKDIR = 'work_dirs'
    GEN_CONFIG_DIR = 'configs/generated'

    parser = argparse.ArgumentParser() # creates an ArgumentParser object, which will be used to define and parse command-line arguments.
    group = parser.add_mutually_exclusive_group(required=True)  # creates a mutually exclusive group of arguments. In this group, you can specify that the user can only provide one of the options, but not both. The required=True argument means that the user must provide one of the mutually exclusive options.
    group.add_argument('--exp', type=int, default=expId, help='Experiment id as defined in experiment.py') # adds an argument to the mutually exclusive group. It specifies that the user can provide the --exp option, which should be an integer. If the user doesn't provide this option, it will default to the value of expId. The help parameter provides a description of what this argument does.
    group.add_argument('--config', default=config_file, help='Path to config file', ) 
    parser.add_argument('--machine', type=str, default=machine_name, help='Name of the machine')
    parser.add_argument('--exp_root', type=str, default=None, help='Root folder to save all EDAPS experimental results') # parses the command-line arguments provided by the user based on the definitions you provided above. The parsed arguments are stored in the args object
    parser.add_argument('--exp_sub', type=str, default=None, help='sub folder to save experimental results benong to a spefic experiment Id')
    args = parser.parse_args()
    assert (args.config is None) != (args.exp is None), 'Either config or exp has to be defined.'

    cfgs, config_files = [], []
    # Training with Predefined Config
    if args.config is not None:
        print(f'training with predefined config : {args.config}')
        cfg = Config.fromfile(args.config)
        # Specify Name and Work Directory
        exp_name = f'{args.machine}-exp{cfg["exp"]:05d}' # local-exp00001 , 1 from given confiquration file
        unique_name = f'{datetime.now().strftime("%y%m%d_%H%M")}_{cfg["name"]}_{str(uuid.uuid4())[:5]}'
        # setting paths for panoptic evaluation
        panop_eval_folder = os.path.join(cfg['exp_root'], cfg['exp_sub'], WORKDIR, exp_name, unique_name, 'panoptic_eval')
        panop_eval_temp_folder = create_panop_eval_folders(panop_eval_folder)
        panop_eval_outdir = os.path.join(panop_eval_temp_folder, 'visuals')
        child_cfg = {
            '_base_': args.config.replace('configs', '../..'),
            'name': unique_name,
            'work_dir': os.path.join(cfg['exp_root'], cfg['exp_sub'], WORKDIR, exp_name, unique_name),
            'git_rev': get_git_hash(),
            'evaluation': {
                            'panop_eval_folder': panop_eval_folder,
                            'panop_eval_temp_folder': panop_eval_temp_folder,
                            'debug': cfg['debug'],
                            'out_dir': panop_eval_outdir,
                            },
        }
        cfg_out_file = f"{GEN_CONFIG_DIR}/{exp_name}/{child_cfg['name']}.json"
        os.makedirs(os.path.dirname(cfg_out_file), exist_ok=True)
        assert not os.path.isfile(cfg_out_file)
        with open(cfg_out_file, 'w') as of:
            json.dump(child_cfg, of, indent=4)
        config_files.append(cfg_out_file)
        cfgs.append(cfg)

    # Training with Generated Configs from experiments.py
    if args.exp is not None:
        if args.exp in [100, 101, 102]:
            # Experiments belongs to M-Dec-BU or S-Net models (Table 5 in the main paper).
            cfgs = generate_experiment_cfgs_bottomup(args.exp, args.machine)
        elif args.exp in [1, 2, 3, 4, 50, 51, 52, 53, 6, 7, 8, 9, 10]:
            # Experiments belongs to EDAPS (M-Dec-TD) mdoels.
            cfgs = generate_experiment_cfgs(args.exp, args.machine)
        else:
            raise NotImplementedError(f"Do not find implementation for experiment id : {args.exp} !!")
        for i, cfg in enumerate(cfgs):
            machine = cfg['machine']
            exp_name = '{}-exp{:05d}'.format(cfg['machine'], args.exp)
            cfg['name'] = f'{datetime.now().strftime("%y%m%d_%H%M")}_{cfg["name"]}_{str(uuid.uuid4())[:5]}'
            cfg['work_dir'] = os.path.join(cfg['exp_root'], cfg['exp_sub'], WORKDIR, exp_name, cfg['name'])
            cfg['git_rev'] = get_git_hash()
            cfg['_base_'] = ['../../' + e for e in cfg['_base_']]
            # set configs for panoptic evaluation
            panop_eval_folder = os.path.join(cfg['work_dir'], 'panoptic_eval')
            panop_eval_temp_folder = create_panop_eval_folders(panop_eval_folder)
            cfg['evaluation']['panop_eval_folder'] = panop_eval_folder
            cfg['evaluation']['panop_eval_temp_folder'] = panop_eval_temp_folder
            cfg['evaluation']['debug']=cfg['debug']
            cfg['evaluation']['out_dir'] = os.path.join(panop_eval_temp_folder, 'visuals')
            # write cfg to json file
            cfg_out_file = f"{GEN_CONFIG_DIR}/{exp_name}/{cfg['name']}.json"
            os.makedirs(os.path.dirname(cfg_out_file), exist_ok=True)
            assert not os.path.isfile(cfg_out_file)
            with open(cfg_out_file, 'w') as of:
                json.dump(cfg, of, indent=4)
            config_files.append(cfg_out_file)

    if args.machine == 'local':
        for i, cfg in enumerate(cfgs):
            print('Run job {}'.format(cfg['name']))
            if cfg['exp'] in [100, 101]:
                train.main([config_files[i]])
            elif cfg['exp'] in [1, 2, 3, 4, 50, 51]:
                train_mmdet.main([config_files[i]])
            elif cfg['exp'] in [52, 53, 6, 7, 8, 9, 10]:
                test_mmdet.main([config_files[i]])
            elif cfg['exp'] in [102]:
                test.main([config_files[i]])
            else:
                raise NotImplementedError(f"Do not find implementation for experiment id : {args.exp} !!")
            torch.cuda.empty_cache()
