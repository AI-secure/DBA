

import datetime
import utils.csv_record as csv_record
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import main
import loan_train
import image_train
import config
import random

def train(helper, start_epoch, local_model, target_model, is_poison,agent_name_keys):
    epochs_submit_update_dict={}
    num_samples_dict={}
    if helper.params['type'] == config.TYPE_LOAN:
        epochs_submit_update_dict, num_samples_dict = loan_train.LoanTrain(helper, start_epoch, local_model, target_model, is_poison,agent_name_keys)
    elif helper.params['type'] == config.TYPE_CIFAR \
            or helper.params['type'] == config.TYPE_MNIST \
            or helper.params['type']==config.TYPE_TINYIMAGENET:
        epochs_submit_update_dict, num_samples_dict = image_train.ImageTrain(helper, start_epoch, local_model,
                                                                             target_model, is_poison, agent_name_keys)
    return epochs_submit_update_dict, num_samples_dict
