import numpy as np
import random
import torch
from torch.autograd import Variable
from torch.utils.data.sampler import Sampler


def dict_html(dict_obj, current_time):
    out = ''
    for key, value in dict_obj.items():

        #filter out not needed parts:
        if key in ['poisoning_test', 'test_batch_size', 'discount_size', 'folder_path', 'log_interval',
                   'coefficient_transfer', 'grad_threshold' ]:
            continue

        out += f'<tr><td>{key}</td><td>{value}</td></tr>'
    output = f'<h4>Params for model: {current_time}:</h4><table>{out}</table>'
    return output

