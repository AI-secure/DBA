import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import datetime


class SimpleNet(nn.Module):
    def __init__(self, name=None, created_time=None):
        super(SimpleNet, self).__init__()
        self.created_time = created_time
        self.name=name

    def train_vis(self, vis, epoch, acc, loss=None, eid='main', is_poisoned=False, name=None):
        if name is None:
            name = self.name + '_poisoned' if is_poisoned else self.name
        vis.line(X=np.array([epoch]), Y=np.array([acc]), name=name, win='train_acc_{0}'.format(self.created_time), env=eid,
                                update='append' if vis.win_exists('train_acc_{0}'.format(self.created_time), env=eid) else None,
                                opts=dict(showlegend=True, title='Train Accuracy_{0}'.format(self.created_time),
                                          width=700, height=400))
        if loss is not None:
            vis.line(X=np.array([epoch]), Y=np.array([loss]), name=name, env=eid,
                                     win='train_loss_{0}'.format(self.created_time),
                                     update='append' if vis.win_exists('train_loss_{0}'.format(self.created_time), env=eid) else None,
                                     opts=dict(showlegend=True, title='Train Loss_{0}'.format(self.created_time), width=700, height=400))
        return

    def train_batch_vis(self, vis, epoch, data_len, batch, loss, eid='main', name=None, win='train_batch_loss', is_poisoned=False):
        if name is None:
            name = self.name + '_poisoned' if is_poisoned else self.name
        else:
            name = name + '_poisoned' if is_poisoned else name

        vis.line(X=np.array([(epoch-1)*data_len+batch]), Y=np.array([loss]),
                                 env=eid,
                                 name=f'{name}' if name is not None else self.name, win=f'{win}_{self.created_time}',
                                 update='append' if vis.win_exists(f'{win}_{self.created_time}', env=eid) else None,
                                 opts=dict(showlegend=True, width=700, height=400, title='Train Batch loss_{0}'.format(self.created_time)))
    def track_distance_batch_vis(self,vis, epoch, data_len, batch, distance_to_global_model,eid,name=None,is_poisoned=False):
        x= (epoch-1)*data_len+batch+1

        if name is None:
            name = self.name + '_poisoned' if is_poisoned else self.name
        else:
            name = name + '_poisoned' if is_poisoned else name


        vis.line(Y=np.array([distance_to_global_model]), X=np.array([x]),
                 win=f"global_dist_{self.created_time}",
                 env=eid,
                 name=f'Model_{name}',
                 update='append' if
                 vis.win_exists(f"global_dist_{self.created_time}",
                                env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Distance to Global {self.created_time}",
                           width=700, height=400))
    def weight_vis(self,vis,epoch,weight, eid, name,is_poisoned=False):
        name = str(name) + '_poisoned' if is_poisoned else name
        vis.line(Y=np.array([weight]), X=np.array([epoch]),
                 win=f"Aggregation_Weight_{self.created_time}",
                 env=eid,
                 name=f'Model_{name}',
                 update='append' if
                 vis.win_exists(f"Aggregation_Weight_{self.created_time}",
                                env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Aggregation Weight {self.created_time}",
                           width=700, height=400))

    def alpha_vis(self,vis,epoch,alpha, eid, name,is_poisoned=False):
        name = str(name) + '_poisoned' if is_poisoned else name
        vis.line(Y=np.array([alpha]), X=np.array([epoch]),
                 win=f"FG_Alpha_{self.created_time}",
                 env=eid,
                 name=f'Model_{name}',
                 update='append' if
                 vis.win_exists(f"FG_Alpha_{self.created_time}",
                                env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"FG Alpha {self.created_time}",
                           width=700, height=400))

    def trigger_test_vis(self, vis, epoch, acc, loss, eid, agent_name_key, trigger_name, trigger_value):
        vis.line(Y=np.array([acc]), X=np.array([epoch]),
                 win=f"poison_triggerweight_vis_acc_{self.created_time}",
                 env=eid,
                 name=f'{agent_name_key}_[{trigger_name}]_{trigger_value}',
                 update='append' if vis.win_exists(f"poison_trigger_acc_{self.created_time}",
                                                   env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Backdoor Trigger Test Accuracy_{self.created_time}",
                           width=700, height=400))
        if loss is not None:
            vis.line(Y=np.array([loss]), X=np.array([epoch]),
                     win=f"poison_trigger_loss_{self.created_time}",
                     env=eid,
                     name=f'{agent_name_key}_[{trigger_name}]_{trigger_value}',
                     update='append' if vis.win_exists(f"poison_trigger_loss_{self.created_time}",
                                                       env=eid) else None,
                     opts=dict(showlegend=True,
                               title=f"Backdoor Trigger Test Loss_{self.created_time}",
                               width=700, height=400))

    def trigger_agent_test_vis(self, vis, epoch, acc, loss, eid, name):
        vis.line(Y=np.array([acc]), X=np.array([epoch]),
                 win=f"poison_state_trigger_acc_{self.created_time}",
                 env=eid,
                 name=f'{name}',
                 update='append' if vis.win_exists(f"poison_state_trigger_acc_{self.created_time}",
                                                   env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Backdoor State Trigger Test Accuracy_{self.created_time}",
                           width=700, height=400))
        if loss is not None:
            vis.line(Y=np.array([loss]), X=np.array([epoch]),
                     win=f"poison_state_trigger_loss_{self.created_time}",
                     env=eid,
                     name=f'{name}',
                     update='append' if vis.win_exists(f"poison_state_trigger_loss_{self.created_time}",
                                                       env=eid) else None,
                     opts=dict(showlegend=True,
                               title=f"Backdoor State Trigger Test Loss_{self.created_time}",
                               width=700, height=400))


    def poison_test_vis(self, vis, epoch, acc, loss, eid, agent_name_key):
        name= agent_name_key
        # name= f'Model_{name}'

        vis.line(Y=np.array([acc]), X=np.array([epoch]),
                 win=f"poison_test_acc_{self.created_time}",
                 env=eid,
                 name=name,
                 update='append' if vis.win_exists(f"poison_test_acc_{self.created_time}",
                                                   env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Backdoor Task Accuracy_{self.created_time}",
                           width=700, height=400))
        if loss is not None:
            vis.line(Y=np.array([loss]), X=np.array([epoch]),
                     win=f"poison_loss_acc_{self.created_time}",
                     env=eid,
                     name=name,
                     update='append' if vis.win_exists(f"poison_loss_acc_{self.created_time}",
                                                       env=eid) else None,
                     opts=dict(showlegend=True,
                               title=f"Backdoor Task Test Loss_{self.created_time}",
                               width=700, height=400))

    def additional_test_vis(self, vis, epoch, acc, loss, eid, agent_name_key):
        name = agent_name_key
        vis.line(Y=np.array([acc]), X=np.array([epoch]),
                 win=f"additional_test_acc_{self.created_time}",
                 env=eid,
                 name=name,
                 update='append' if vis.win_exists(f"additional_test_acc_{self.created_time}",
                                                   env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Additional Test Accuracy_{self.created_time}",
                           width=700, height=400))
        if loss is not None:
            vis.line(Y=np.array([loss]), X=np.array([epoch]),
                     win=f"additional_test_loss_{self.created_time}",
                     env=eid,
                     name=name,
                     update='append' if vis.win_exists(f"additional_test_loss_{self.created_time}",
                                                       env=eid) else None,
                     opts=dict(showlegend=True,
                               title=f"Additional Test Loss_{self.created_time}",
                               width=700, height=400))


    def test_vis(self, vis, epoch, acc, loss, eid, agent_name_key):
        name= agent_name_key
        # name= f'Model_{name}'

        vis.line(Y=np.array([acc]), X=np.array([epoch]),
                 win=f"test_acc_{self.created_time}",
                 env=eid,
                 name=name,
                 update='append' if vis.win_exists(f"test_acc_{self.created_time}",
                                                   env=eid) else None,
                 opts=dict(showlegend=True,
                           title=f"Main Task Test Accuracy_{self.created_time}",
                           width=700, height=400))
        if loss is not None:
            vis.line(Y=np.array([loss]), X=np.array([epoch]),
                     win=f"test_loss_{self.created_time}",
                     env=eid,
                     name=name,
                     update='append' if vis.win_exists(f"test_loss_{self.created_time}",
                                                       env=eid) else None,
                     opts=dict(showlegend=True,
                               title=f"Main Task Test Loss_{self.created_time}",
                               width=700, height=400))


    def save_stats(self, epoch, loss, acc):
        self.stats['epoch'].append(epoch)
        self.stats['loss'].append(loss)
        self.stats['acc'].append(acc)

    def copy_params(self, state_dict, coefficient_transfer=100):

        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                shape = param.shape
                #random_tensor = (torch.cuda.FloatTensor(shape).random_(0, 100) <= coefficient_transfer).type(torch.cuda.FloatTensor)
                # negative_tensor = (random_tensor*-1)+1
                # own_state[name].copy_(param)
                own_state[name].copy_(param.clone())




class SimpleMnist(SimpleNet):
    def __init__(self, name=None, created_time=None):
        super(SimpleMnist, self).__init__(name, created_time)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)


    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)