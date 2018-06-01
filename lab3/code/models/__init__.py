import os
import copy

import numpy as np
import misc.utils as utils
import torch

from .ShowTellModel import ShowTellModel
from .OldModel import ShowAttendTellModel, AllImgModel
from .AttModel import *

def setup(opt):
    

    if opt.caption_model == 'show_attend_tell':#part1
        model = ShowAttendTellModel(opt)
    # img is concatenated with word embedding at every time step as the input of lstm
    elif opt.caption_model == 'topdown':#part2
        model = TopDownModel(opt)
    else:
        raise Exception("Caption model not supported: {}".format(opt.caption_model))

    # check compatibility if training is continued from previously saved model
    if vars(opt).get('start_from', None) is not None:
        # check if all necessary files exist 
        assert os.path.isdir(opt.start_from)," %s must be a a path" % opt.start_from
        assert os.path.isfile(os.path.join(opt.start_from,"infos_"+opt.id+".pkl")),"infos.pkl file does not exist in path %s"%opt.start_from
        model.load_state_dict(torch.load(os.path.join(opt.start_from, 'model.pth')))

    return model