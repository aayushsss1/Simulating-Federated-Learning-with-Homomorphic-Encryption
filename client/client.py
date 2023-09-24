import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import Variable
import numpy as np
import utils
import matplotlib.pyplot as plt
import tenseal as ts
import pandas as pd
from localmodel import LogisticRegression, scale_dataset

class Client:
    def __init__(self, name, data_url, enc_file, n_features, iters):
        self.id = name
        self.enc_file = enc_file  # place where clients save encrypted weights
        
        # split data into train and test
        self.X_train, self.Y_train, self.X_test, self.Y_test = self.preprocessing(data_url)
        
        # define local training model
        self.local_model = LogisticRegression(n_features)
        
        # some helpfull stuffs
        self.decide_vectorized = np.vectorize(self.decide)
        self.to_percent = lambda x: '{:.2f}%'.format(x)
        self.num_epochs = iters
        self.accuracies = []
        self.losses = []
        
    def preprocessing(self, data_url):
        df = pd.read_csv(data_url)
        # Replace "M" with 1 and "B" with 0 at "diagnostic" column
        df["diagnostic"] = (df["diagnostic"] == "M").astype(int)
        
        # split dataframe to train and test df
        df_train, df_test = np.split(df.sample(frac=1), [int(0.8 * len(df))])
        
        # scaling and convert to tensor context
        train, X_train, Y_train = scale_dataset(df_train, True)
        test , X_test , Y_test  = scale_dataset(df_test , False)
        return X_train, Y_train, X_test, Y_test
    
    def decide(self, y):
        return 1. if y >= 0.5 else 0.
    
    def compute_accuracy(self, input, output):
        prediction = self.local_model(input).data.numpy()[:, 0]
        n_samples = prediction.shape[0] + 0.
        prediction = self.decide_vectorized(prediction)
        equal = prediction == output.data.numpy()
        return 100. * equal.sum() / n_samples
    
    def local_training(self, debug=True):
        n_samples, _ = self.X_train.shape

        # define criterion function and set up optimizer
        criterion = torch.nn.BCELoss(reduction='mean')
        optimizer = torch.optim.SGD(self.local_model.parameters(), lr=0.01)  

        # main process
        for epoch in range(self.num_epochs):  
            optimizer.zero_grad()
            #### Compute outputs ####
            prediction = self.local_model(self.X_train)

            #### Compute gradients ####
            loss = criterion(prediction.squeeze(), self.Y_train)
            loss.backward()

            #### Update weights #### 
            optimizer.step()

            # compute accuracy and loss
            train_acc = self.compute_accuracy(self.X_train, self.Y_train)
            train_loss = loss.item()
                
            self.losses.append(train_loss)
            self.accuracies.append(train_acc)
        
            #### Logging ####
            if debug and (epoch + 1)%50 == 0:
                print('[LOG] Epoch: %05d' % (epoch + 1), end="")
                print('    | Train ACC: %s' % self.to_percent(train_acc), end="")
                print('    | Loss: %.3f' % train_loss)
    
    def encrypted_model_params(self):
        model_weights = self.local_model.linear.weight.data.squeeze().tolist()
        model_bias    = self.local_model.linear.bias.data.squeeze().tolist()
        model_params  = model_weights + [model_bias]
        context = ts.context_from(utils.read_data("keys/secret.txt"))
        encrypt_weights = ts.ckks_vector(context, model_params)
        utils.write_data(self.enc_file, encrypt_weights.serialize())