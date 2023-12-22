# This script contains the helper functions you will be using for this assignment

import os
import random

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BassetDataset(Dataset):
    """
    BassetDataset class taken with permission from Dr. Ahmad Pesaranghader

    We have already processed the data in HDF5 format: er.h5
    See https://www.h5py.org/ for details of the Python package used

    We used the same data processing pipeline as the paper.
    You can find the code here: https://github.com/davek44/Basset
    """

    # Initializes the BassetDataset
    def __init__(self, path='./data/', f5name='er.h5', split='train', transform=None):
        """
        Args:
            :param path: path to HDF5 file
            :param f5name: HDF5 file name
            :param split: split that we are interested to work with
            :param transform (callable, optional): Optional transform to be applied on a sample
        """

        self.split = split

        split_dict = {'train': ['train_in', 'train_out'],
                      'test': ['test_in', 'test_out'],
                      'valid': ['valid_in', 'valid_out']}

        assert self.split in split_dict, "'split' argument can be only defined as 'train', 'valid' or 'test'"

        # Open hdf5 file where one-hoted data are stored
        self.dataset = h5py.File(os.path.join(path, f5name.format(self.split)), 'r')

        # Keeping track of the names of the target labels
        self.target_labels = self.dataset['target_labels']

        # Get the list of volumes
        self.inputs = self.dataset[split_dict[split][0]]
        self.outputs = self.dataset[split_dict[split][1]]

        self.ids = list(range(len(self.inputs)))
        if self.split == 'test':
            self.id_vars = np.char.decode(self.dataset['test_headers'])

    def __getitem__(self, i):
        """
        Returns the sequence and the target at index i

        Notes:
        * The data is stored as float16, however, your model will expect float32.
          Do the type conversion here!
        * Pay attention to the output shape of the data.
          Change it to match what the model is expecting
          hint: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        * The target must also be converted to float32
        * When in doubt, look at the output of __getitem__ !
        """

        idx = self.ids[i]

        # Sequence & Target
        output = {'sequence': None, 'target': None}

        # WRITE CODE 
        # print(self.inputs[idx].shape)
        # print(self.inputs[idx].reshape(1,600,4).shape)
        output['sequence'] = torch.from_numpy(self.inputs[idx].astype(np.float32)).permute(1,2,0) #(1,2,0) cuz we switch from (4, 1, 600) to (1, 600, 4)
        output['target'] = torch.from_numpy(self.outputs[idx].astype(np.float32))
        return output

    def __len__(self):
        # WRITE CODE HERE
        return len(self.inputs)

    def get_seq_len(self):
        """
        Answer to Q1 part 2
        """
        # WRITE CODE HERE
        # print(len(self.__getitem__(0)['sequence'][0]))
        return  len(self.__getitem__(0)['sequence'][0])

    def is_equivalent(self):
        """
        Answer to Q1 part 3
        """
        # WRITE CODE HERE
        t_v = self.get_seq_len()
        return self.__getitem__(0)['sequence'].shape==(1,t_v, 4)


class Basset(nn.Module):
    """
    Basset model
    Architecture specifications can be found in the supplementary material
    You will also need to use some Convolution Arithmetic
    """

    def __init__(self):
        super(Basset, self).__init__()

        self.dropout = 0.3  # should be float
        self.drop = nn.Dropout(self.dropout)
        self.num_cell_types = 164

        self.conv1 = nn.Conv2d(1, 300, (19, 4), stride=(1, 1), padding=(9, 0))
        self.conv2 = nn.Conv2d(300, 200, (11, 1), stride=(1, 1), padding=(5, 0))
        self.conv3 = nn.Conv2d(200, 200, (7, 1), stride=(1, 1), padding=(4, 0))

        self.bn1 = nn.BatchNorm2d(300)
        self.bn2 = nn.BatchNorm2d(200)
        self.bn3 = nn.BatchNorm2d(200)
        self.maxpool1 = nn.MaxPool2d((3, 1))
        self.maxpool2 = nn.MaxPool2d((4, 1))
        self.maxpool3 = nn.MaxPool2d((4, 1))

        self.fc1 = nn.Linear(13*200, 1000)
        self.bn4 = nn.BatchNorm1d(1000)

        self.fc2 = nn.Linear(1000, 1000)
        self.bn5 = nn.BatchNorm1d(1000)

        self.fc3 = nn.Linear(1000, self.num_cell_types)

    def forward(self, x):
        """
        Compute forward pass for the model.
        nn.Module will automatically create the `.backward` method!

        Note:
            * You will have to use torch's functional interface to 
              complete the forward method as it appears in the supplementary material
            * There are additional batch norm layers defined in `__init__`
              which you will want to use on your fully connected layers
        """
        #From https://genome.cshlp.org/content/suppl/2016/06/10/gr.200535.115.DC1/Supplementary_Figures.pdf, P.14
        # WRITE CODE HERE
        #print(" Shape of X at the beginning : ", x.shape)
       
        #ConvNet
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x= self.maxpool1(x)

        x= self.conv2(x)
        x= self.bn2(x)
        x= F.relu(x)
        x= self.maxpool2(x)

        x= self.conv3(x)
        x=self.bn3(x)
        x= F.relu(x)
        x = self.maxpool3(x)

        #print("Shape of X after the ConvNet (before flatten) : ", x.shape)
        #FLatten !!
        
        x= torch.flatten(x, start_dim=1)

        #print("Shape of X after the ConvNet (after flatten) : ", x.shape)
        #Fully Connected network
        x= self.fc1(x)
        x = self.bn4(x)
        x= self.drop(x)
        x= F.relu(x)
        
        x= self.fc2(x)
        x = self.bn5(x)
        x= self.drop(x)
        x= F.relu(x)

        #output structure
        x = self.fc3(x)
        return x


def compute_fpr_tpr(y_true, y_pred):
    """
    Computes the False Positive Rate and True Positive Rate
    Args:
        :param y_true: groundtruth labels (np.array of ints [0 or 1])
        :param y_pred: model decisions (np.array of ints [0 or 1])

    :Return: dict with tpr, fpr (values are floats)
    """
    output = {'fpr': 0., 'tpr': 0.}

    # WRITE CODE HERE
    # from https://stackoverflow.com/questions/61321778/how-to-calculate-tpr-and-fpr-in-python-without-using-sklearn
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tp = np.sum((y_pred == 1) & (y_true == 1))

    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    output['fpr'] = fp / (fp + tn)
    output['tpr'] = tp / (tp + fn)
    return output


def compute_fpr_tpr_dumb_model():
    """
    Simulates a dumb model and computes the False Positive Rate and True Positive Rate

    :Return: dict with tpr_list, fpr_list.
             These lists contain the tpr and fpr for different thresholds (k)
             fpr and tpr values in the lists should be floats
             Order the lists such that:
                 output['fpr_list'][0] corresponds to k=0.
                 output['fpr_list'][1] corresponds to k=0.05
                 ...
                 output['fpr_list'][-1] corresponds to k=0.95

            Do the same for output['tpr_list']
             
    """
    output = {'fpr_list': [], 'tpr_list': []}

    # WRITE CODE HERE
    #simulate a dumb model on 1000 samples
    s=1000
    target = np.random.randint(2, size=s)
    pred = np.random.uniform(low=0,high=1,size=s)
    r = np.arange(start=0,stop=1,step=0.05,dtype=float) # threshold range
    #  print(r)
    #Compute tpr and fpr
    for k in r:
        pred_thresh = np.where(pred >=k,1,0)
        #print(pred_thresh.dtype)
        out = compute_fpr_tpr(target, pred_thresh)
        output['fpr_list'].append(out['fpr'])
        output['tpr_list'].append(out['tpr'])
    

    return output


def compute_fpr_tpr_smart_model():
    """
    Simulates a smart model and computes the False Positive Rate and True Positive Rate

    :Return: dict with tpr_list, fpr_list.
             These lists contain the tpr and fpr for different thresholds (k)
             fpr and tpr values in the lists should be floats
             Order the lists such that:
                 output['fpr_list'][0] corresponds to k=0.
                 output['fpr_list'][1] corresponds to k=0.05
                 ...
                 output['fpr_list'][-1] corresponds to k=0.95

            Do the same for output['tpr_list']
    """
    output = {'fpr_list': [], 'tpr_list': []}

    # WRITE CODE HERE
    #simulate smart model on 1000 range
    s=1000
    target = np.random.randint(2, size=s)
    positive_case = np.random.uniform(low=0.4,high=1,size=s)*(target) #mult to get only positive
    negative_case = np.random.uniform(low=0,high=0.6,size=s) *(1-target) #same for negative
    pred = positive_case+negative_case #merge

    r = np.arange(start=0,stop=1,step=0.05,dtype=float) # threshold range
    for k in r:
        pred_thresh = np.where(pred >=k,1,0)
        #print(pred_thresh.dtype)
        out = compute_fpr_tpr(target, pred_thresh)
        output['fpr_list'].append(out['fpr'])
        output['tpr_list'].append(out['tpr'])
    
    return output


def compute_auc_both_models():
    """
    Simulates a dumb model and a smart model and computes the AUC of both

    :Return: dict with auc_dumb_model, auc_smart_model.
             These contain the AUC for both models
             auc values in the lists should be floats
    """
    output = {'auc_dumb_model': 0., 'auc_smart_model': 0.}

    # WRITE CODE HERE
    s=1000
    target = np.random.randint(2, size=s)

    dumb_values = np.random.uniform(low=0,high=1,size=s)
    positive_case = np.random.uniform(low=0.4,high=1,size=s)*(target) #mult to get only positive
    negative_case = np.random.uniform(low=0,high=0.6,size=s) *(1-target) #same for negative
    smart_values= positive_case+negative_case
    
    output['auc_dumb_model'] = compute_auc(target,dumb_values)['auc']
    output['auc_smart_model'] = compute_auc(target,smart_values)['auc']
    return output


def compute_auc_untrained_model(model, dataloader, device):
    """
    Computes the AUC of your input model

    Dont forget to re-apply your output activation!

    :Return: dict with auc_dumb_model, auc_smart_model.
             These contain the AUC for both models
             auc values should be floats

    Make sure this function works with arbitrarily small dataset sizes!
    """
    output = {'auc': 0.}

    # WRITE CODE HERE
    # print(type(model))
    # print(type(dataloader))
    # print(type(device))

    model = model.to(device)
    model.eval()
    y_pred = torch.Tensor().to(device)
    y_true = torch.Tensor().to(device)

    #get true and pred from dataloader
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            sequence = batch['sequence'].to(device) #to device if we are in CPU/GPU
            targets = batch['target'].to(device) #to device if we are in CPU/GPU
            act = torch.sigmoid(model(sequence))
            y_pred = torch.cat((y_pred, act))
            y_true = torch.cat((y_true,targets))

    #apply output activation 
    
    # print(type(act))
    # print(type(targets))
    # AHHHHH : https://stackoverflow.com/questions/49768306/pytorch-tensor-to-numpy-array
    output['auc'] = compute_auc(y_true.detach().cpu().numpy().flatten(),y_pred.detach().cpu().numpy().flatten())['auc']
    return output


def compute_auc(y_true, y_model):
    """
    Computes area under the ROC curve
    auc returned should be float
    Args:
        :param y_true: groundtruth labels (np.array of ints [0 or 1])
        :param y_model: model outputs (np.array of float32 in [0, 1])

    Note: if you set y_model as the output of solution.Basset, 
    you need to transform it first!
    """
    output = {'auc': 0.}

    # WRITE CODE HERE
    #compute fpr and tpr the same way as before
    target = y_true
    pred = y_model
    fpr_tpr_list  = {'fpr_list': [], 'tpr_list': []}
    r = np.arange(start=0,stop=1,step=0.05,dtype=float) # threshold range
    for k in r:
        pred_thresh = np.where(pred >=k,1,0)
        #print(pred_thresh.dtype)
        out = compute_fpr_tpr(target, pred_thresh)
        fpr_tpr_list['fpr_list'].append(out['fpr'])
        fpr_tpr_list['tpr_list'].append(out['tpr'])

    fprs= fpr_tpr_list['fpr_list']
    tprs= fpr_tpr_list['tpr_list']
    #print(type(fprs))
    #np.trapz doesn't work (F*CK IT) so I'll use left and right Riemann's sum : (https://www.khanacademy.org/math/ap-calculus-ab/ab-integration-new/ab-6-2/a/left-and-right-riemann-sums) 
    #Compute left Riemann sum
    assert(len(fprs)==len(tprs))
    #get left Riemann
    left=0.
    for i in range(0,len(fprs)-1):
        delta= np.abs(fprs[i] - fprs[i+1])
        height = tprs[i]
        #print("Width : " + width + " Heigth : "+ height )
        left+=delta*height

    #print("left : ",left)
    #get right Riemann
    right=0.
    for i in range(1,len(fprs)):
        delta= np.abs(fprs[i-1] - fprs[i])
        height = tprs[i]
        #print("Width : " + width + " Heigth : "+ height )
        right+=delta*height

    #print("right : ",right)
    output['auc'] = np.abs((left+right))/2 #abs cuz we have some negative values
    #print("Output : ", output['auc'])
    return output


def get_critereon():
    """
    Picks the appropriate loss function for our task
    criterion should be subclass of torch.nn
    """

    # WRITE CODE HERE
    critereon = nn.BCEWithLogitsLoss()
    return critereon


def train_loop(model, train_dataloader, device, optimizer, criterion):
    """
    One Iteration across the training set
    Args:
        :param model: solution.Basset()
        :param train_dataloader: torch.utils.data.DataLoader
                                 Where the dataset is solution.BassetDataset
        :param device: torch.device
        :param optimizer: torch.optim
        :param critereon: torch.nn (output of get_critereon)

    :Returns: output dict with keys: total_score, total_loss
    values of each should be floats
    (if you want to display losses and/or scores within the loop, you may print them to screen)

    Make sure your loop works with arbitrarily small dataset sizes!

    Note: you don’t need to compute the score after each training iteration.
    If you do this, your training loop will be really slow!
    You should instead compute it every 50 or so iterations and aggregate ...
    """

    output = {'total_score': 0.,
              'total_loss': 0.}

    # WRITE CODE HERE
    cuda = torch.cuda.is_available()
    if cuda:
        print('CUDA is available')
    else:
        print('CUDA is not available')
    train_loss  =0.

    y_pred = torch.Tensor().to(device)
    y_true = torch.Tensor().to(device)

    model.train() # call train
    #loop through dataset
    for i, batch in enumerate(train_dataloader):
        optimizer.zero_grad()

        sequence = batch['sequence'].to(device) #data (features)
        targets = batch['target'].to(device)#labels

        #output of declared layers
        out = model(sequence)
        assert(len(out)==len(targets))
        loss = criterion(out,targets)
        
        #accumulate gradient (backprop)
        loss.backward()

        optimizer.step() #update

        train_loss+=loss.item() #item get the loss value
        y_true = torch.cat((y_true, targets)) #labels

        #don't forget sigmoid
        out = torch.sigmoid(out)
        y_pred =torch.cat((y_pred,out))

    # print(type(y_pred))
    # print(type(y_true))
    # print(type(train_loss))

    output['total_score'] = compute_auc(y_true.detach().cpu().numpy().flatten(),y_pred.detach().cpu().numpy().flatten())['auc']
    output['total_loss'] = train_loss

    return output['total_score'], output['total_loss']


def valid_loop(model, valid_dataloader, device, optimizer, criterion):
    """
    One Iteration across the validation set
    Args:
        :param model: solution.Basset()
        :param valid_dataloader: torch.utils.data.DataLoader
                                 Where the dataset is solution.BassetDataset
        :param device: torch.device
        :param optimizer: torch.optim
        :param critereon: torch.nn (output of get_critereon)

    :Returns: output dict with keys: total_score, total_loss
    values of each should be floats
    (if you want to display losses and/or scores within the loop, you may print them to screen)

    Make sure your loop works with arbitrarily small dataset sizes!
    """

    output = {'total_score': 0.,
              'total_loss': 0.}

    # WRITE CODE HERE
    valid_loss = 0.
    y_pred = torch.Tensor().to(device)
    y_true = torch.Tensor().to(device)

    model.eval() # call eval
    for i, batch in enumerate(valid_dataloader):
        optimizer.zero_grad()

        sequence = batch['sequence'].to(device) #data (features)
        targets = batch['target'].to(device)#labels

        #output of declared layers
        out = model(sequence)
        assert(len(out)==len(targets))
        loss = criterion(out,targets)
        
        #accumulate gradient (backprop)
        loss.backward()

        optimizer.step() #update

        valid_loss+=loss.item() #item get the loss value
        y_true = torch.cat((y_true, targets)) #labels

        #don't forget sigmoid
        out = torch.sigmoid(out)
        y_pred =torch.cat((y_pred,out))

    output['total_score'] = compute_auc(y_true.detach().cpu().numpy().flatten(),y_pred.detach().cpu().numpy().flatten())['auc']
    output['total_loss'] = valid_loss
    return output['total_score'], output['total_loss']
