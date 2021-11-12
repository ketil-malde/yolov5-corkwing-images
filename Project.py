#!/usr/bin/python3

import os
import os.path
import sys

# model='--branch standardize git@github.com:ketil-malde/Pytorch-UNet'
# dataset='git@github.com:ketil-malde/blue-mussel-drone-testdata'

model='git@github.com:ketil-malde/yolov5'
dataset = 'git@github.com:ketil-malde/corkwing-data'

defaultconf = { 'hyper': 'hyp.yaml',
                'weights': 'yolov5m.pt',
                'dataset': 'dataset.yaml',
                'annotations': ['Nest1_finished', '16_Nest2', 'Nest3', 'Nest4']
            }
# 16_Nest2.xml  Nest1_finished.xml  Nest1.xml  Nest3.xml  Nest4.xml  Under1_finished.xml  Under1.xml  Under2.xml

class Project:

    def __init__(self, conf):
        self.m = None
        self.d = None
        self.config = conf

    def setup(self):
        '''Get data and model repositories'''
        os.system(f'git clone {model}')
        modelpath = os.path.basename(model)
        sys.path.insert(1, modelpath)
        import Model
        self.m = Model.Model(self.config, modelpath)
        sys.path.remove(modelpath)

        os.system(f'git clone {dataset}')
        datapath = os.path.basename(dataset)
        sys.path.insert(1, datapath)
        import Data
        self.d = Data.Data(self.config, datapath)
        sys.path.remove(datapath)

    def get_data(self):
        '''Download and verify data'''
        self.d.get()
        self.d.validate()

    def build_model(self):
        '''Build the model and check it'''
        self.m.build()
        self.m.check()

    def train_model(self):
        '''Train the model'''
        self.m.train()

config = {
    # Configuration items here
    }

if __name__ == '__main__':
    # run training
    p = Project(defaultconf)
    p.setup()
    p.get_data()
    p.build_model()
    p.train_model()

    print('Training completed')
