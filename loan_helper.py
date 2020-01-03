from collections import defaultdict
import config
import torch
import torch.utils.data
import datetime
from helper import Helper
import random
import logging
from torchvision import datasets, transforms
import numpy as np

from models.loan_model import LoanNet
import csv

import os

import pandas as pd

import torch
import torch.utils.data as data
from sklearn.model_selection import train_test_split
import os


import yaml
logger = logging.getLogger("logger")
POISONED_PARTICIPANT_POS = 0

class StateHelper():
    def __init__(self, params):
        self.params= params
        self.name=""

    def load_data(self, filename='./data/loan/loan_IA.csv'):
        logger.info('Loading data')

        self.all_dataset = LoanDataset(filename)

    def get_trainloader(self):

        self.all_dataset.SetIsTrain(True)
        train_loader = torch.utils.data.DataLoader(self.all_dataset, batch_size=self.params['batch_size'],
                                                   shuffle=True)

        return train_loader

    def get_testloader(self):

        self.all_dataset.SetIsTrain(False)
        test_loader = torch.utils.data.DataLoader(self.all_dataset,
                                                  batch_size=self.params['test_batch_size'],
                                                  shuffle=False)

        return test_loader

    def get_poison_trainloader(self):
        self.all_dataset.SetIsTrain(True)

        return torch.utils.data.DataLoader(self.all_dataset,
                                           batch_size=self.params['batch_size'],
                                           shuffle=True)

    def get_poison_testloader(self):

        self.all_dataset.SetIsTrain(False)

        return torch.utils.data.DataLoader(self.all_dataset,
                                           batch_size=self.params['test_batch_size'],
                                           shuffle=False)

    def get_batch(self, train_data, bptt, evaluation=False):
        data, target = bptt
        data = data.float().to(config.device)
        target = target.long().to(config.device)
        if evaluation:
            data.requires_grad_(False)
            target.requires_grad_(False)
        return data, target


class LoanHelper(Helper):
    def poison(self):
        return

    def create_model(self):
        local_model = LoanNet(name='Local',
                               created_time=self.params['current_time'])
        local_model=local_model.to(config.device)

        target_model = LoanNet(name='Target',
                                created_time=self.params['current_time'])
        target_model=target_model.to(config.device)

        if self.params['resumed_model']:
            if torch.cuda.is_available():
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}")
            else:
                loaded_params = torch.load(f"saved_models/{self.params['resumed_model_name']}",
                                           map_location='cpu')
            target_model.load_state_dict(loaded_params['state_dict'])
            self.start_epoch = loaded_params['epoch']+1
            self.params['lr'] = loaded_params.get('lr', self.params['lr'])
            logger.info(f"Loaded parameters from saved model: LR is"
                        f" {self.params['lr']} and current epoch is {self.start_epoch}")
        else:
            self.start_epoch = 1

        self.local_model = local_model
        self.target_model = target_model

    def load_data(self,params_loaded):
        self.statehelper_dic ={}
        self.allStateHelperList=[]
        self.participants_list=[]
        self.advasarial_namelist=params_loaded['adversary_list']
        self.benign_namelist = []
        self.feature_dict = dict()

        filepath_prefix='./data/loan/'
        all_userfilename_list = os.listdir(filepath_prefix)
        for j in range(0,len(all_userfilename_list)):
            user_filename = all_userfilename_list[j]
            state_name = user_filename[5:7]
            helper = StateHelper(params=params_loaded)
            file_path = filepath_prefix+ user_filename
            helper.load_data(file_path)
            self.allStateHelperList.append(helper)
            helper.name = state_name
            self.statehelper_dic[state_name] = helper
            if j==0:
                for k in range(0,len(helper.all_dataset.data_column_name)):
                    self.feature_dict[helper.all_dataset.data_column_name[k]]=k

        for j in range(0, params_loaded['number_of_total_participants']):
            if j >= len(all_userfilename_list):
                break
            user_filename = all_userfilename_list[j]
            state_name = user_filename[5:7]
            if state_name not in self.advasarial_namelist:
                self.benign_namelist.append(state_name)

        if params_loaded['is_random_namelist']==False:
            self.participants_list = params_loaded['participants_namelist']
        else:
            self.participants_list= self.benign_namelist+ self.advasarial_namelist


class LoanDataset(data.Dataset):
    # label from 0 ~ 8
    # ['Current', 'Fully Paid', 'Late (31-120 days)', 'In Grace Period', 'Charged Off',
    # 'Late (16-30 days)', 'Default', 'Does not meet the credit policy. Status:Fully Paid',
    # 'Does not meet the credit policy. Status:Charged Off']

    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.train = True
        self.df = pd.read_csv(csv_file)
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        loans_df = self.df.copy()
        x_feature = list(loans_df.columns)
        x_feature.remove('loan_status')
        x_val = loans_df[x_feature]
        y_val = loans_df['loan_status']
        # x_val.head()
        y_val=y_val.astype('int')
        x_train, x_test, y_train, y_test = train_test_split(x_val, y_val, test_size=0.2, random_state=42)
        self.data_column_name = x_train.columns.values.tolist() # list
        self.label_column_name= x_test.columns.values.tolist()
        self.train_data = x_train.values # numpy array
        self.test_data = x_test.values

        self.train_labels = y_train.values
        self.test_labels = y_test.values

        print(csv_file, "train", len(self.train_data),"test",len(self.test_data))

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            data, label = self.train_data[index], self.train_labels[index]
        else:
            data, label = self.test_data[index], self.test_labels[index]

        return data, label

    def SetIsTrain(self,isTrain):
        self.train =isTrain

    def getPortion(self,loan_status=0):
        train_count= 0
        test_count=0
        for i in range(0,len(self.train_labels)):
            if self.train_labels[i]==loan_status:
                train_count+=1
        for i in range(0,len(self.test_labels)):
            if self.test_labels[i]==loan_status:
                test_count+=1
        return (train_count+test_count)/ (len(self.train_labels)+len(self.test_labels)), \
               train_count/len(self.train_labels), test_count/len(self.test_labels)

if __name__ == '__main__':

    with open(f'./utils/loan_params.yaml', 'r') as f:
        params_loaded = yaml.load(f)
    current_time = datetime.datetime.now().strftime('%b.%d_%H.%M.%S')

    helper = LoanHelper(current_time=current_time, params=params_loaded,
                        name=params_loaded.get('name', 'loan'))
    helper.load_data(params_loaded)
    state_keys = list(helper.statehelper_dic.keys())
    for i in range(0,len(state_keys)):
        state_helper = helper.statehelper_dic[state_keys[i]]
        data_source = state_helper.get_trainloader()
        data_iterator = data_source
        count= 0
        for batch_id, batch in enumerate(data_iterator):
            count +=1
        print(state_keys[i], "train batch num",count)
        break

