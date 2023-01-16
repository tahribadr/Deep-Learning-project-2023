import numpy as np
import torch 
from torch import nn 
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import decimal 

decimal.getcontext().prec = 6


class seqRNN(nn.Module):
    
    def __init__(self, input_size, max_len, batch_size, n_outputs, 
                 n_hidden=64, 
                 n_layers=1,
                 drop_prob=0, 
                 lr=0.0001, 
                 bidirectional=False, 
                 activation = None, 
                 cell = 'lstm',
                 all_hidden = None,
                 split_dense = True,
                 task="classif",
                 train_init = False):


        super().__init__()
        self.drop_prob = drop_prob
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lr = lr
        self.input_size = input_size
        self.max_len = max_len

        #here, we distinguish between the batch size that we initalize the hidden states with, 
        #and the working batch size with wich we go through forward function with
        #this distinction allows us to go forwards the model with a different batch size, without 
        #losing the init batch size that we trained the initial hidden states on
        self.init_batch_size = batch_size 
        self.batch_size = batch_size 
        
        self.bidirectional = bidirectional
        self.D = 2 if self.bidirectional == True else 1
        self.n_outputs = n_outputs
        print("self outputs ", self.n_outputs)
        self.train_on_gpu = torch.cuda.is_available()
        self.activation = activation
        self.cell = cell
        self.task = task
        self.train_init = train_init

        #this attribute means we will feed all hidden vectors from all timesteps to the dense layer
        self.all_hidden = all_hidden


        #this attribute means we will split the dense layer into two fully connected layers
        self.split_dense = split_dense
        
        ## define the LSTM
        #input size is of lenghth 10 bcz data has been hot encoded into 10 features with 1 being the char it is 
        
        if self.cell == 'lstm' :
            self.lstm = nn.LSTM(self.input_size, self.n_hidden, self.n_layers, 
                                dropout=self.drop_prob, batch_first=True, bidirectional=self.bidirectional)
        elif self.cell == 'gru' :
            self.gru = nn.GRU(self.input_size, self.n_hidden, self.n_layers, 
                                dropout=self.drop_prob, batch_first=True, bidirectional=self.bidirectional)
        elif self.cell == 'ellman' :
            self.rnn = nn.RNN(self.input_size, self.n_hidden, self.n_layers, 
                                dropout=self.drop_prob, batch_first=True, bidirectional=self.bidirectional)

        #  define a dropout layer
        self.dropout = nn.Dropout(drop_prob)
        
        #  define the final, fully-connected output layer

        if self.all_hidden and self.split_dense: 
            self.fc1 = nn.Linear(self.max_len*self.n_hidden*self.D, self.max_len*self.n_hidden*self.D//2)
            self.fc2 = nn.Linear(self.max_len*self.n_hidden*self.D//2, self.n_outputs)
        

        elif self.all_hidden and (not self.split_dense):
            self.fc1 = nn.Linear(self.max_len*self.n_hidden*self.D, self.n_outputs)
        
        elif (not self.all_hidden) and self.split_dense:
            #print("fc dimensions: ", self.n_hidden*self.D, self.n_hidden*self.D//2, self.n_outputs)
            self.fc1 = nn.Linear(self.n_hidden*self.D, self.n_hidden*self.D//2)
            self.fc2 = nn.Linear(self.n_hidden*self.D//2, self.n_outputs)

        elif (not self.all_hidden) and (not self.split_dense):
            self.fc1 = nn.Linear(self.n_hidden*self.D, self.n_outputs)



    def forward(self, x, hidden):
        ''' Forward pass through the network. 
            These inputs are x, and the hidden/cell state `hidden`. '''
        
        ## Get the outputs and the new hidden state from the recurrent cell 
        if self.cell == 'lstm' :
            r_output, hidden = self.lstm(x, hidden)
        elif self.cell == 'gru' :
            r_output, hidden = self.gru(x, hidden) 
        elif self.cell == 'ellman' :
            r_output, hidden = self.rnn(x, hidden)

        
        if self.all_hidden:
            #  pass through a dropout layer
            drop_out = self.dropout(r_output)
            #contiguous creates a new tensor with appropriate shape instead of sharing memory with original tensor in 
            #non-contiguous way
            drop_out = drop_out.contiguous().view(-1, self.max_len*self.n_hidden*self.D)
        else:
            r_output = r_output[:,-1,:]
            drop_out = self.dropout(r_output)
            drop_out = drop_out.contiguous().view(-1, self.n_hidden*self.D)
        # put x through the fully-connected layer, we get scores out
        out1 = self.fc1(drop_out)

        if self.split_dense :
            out2 = self.fc2(out1)

            if self.activation == None:
                return out2, hidden
            else:
                proba = self.activation(out2, dim=-1 )

        elif not self.split_dense :
            proba = self.activation(out1, dim=-1 )
                
        # return the final output and the hidden state
        return proba, hidden
    
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        
        #if we initalize with a different batch size than previously declared, 
        #we change the batch size of the entire object and work with the new batch_size
        if batch_size != self.init_batch_size:
            self.init_batch_size = batch_size
            self.batch_size = batch_size
            print("initalizing the hidden states was done using a new batch size "
                  +"than previously declared, the new batch size will become the object batch size")

        weight = next(self.parameters()).data # we use this to create hidden tensors with same data type and same device         
                                             # as weight tensors, using weight.new(shape)
        
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        if self.cell == 'lstm' :
            # print("i got in lstm")
            if (self.train_on_gpu):
                self.h0 =  weight.new(self.n_layers*self.D, batch_size, self.n_hidden).zero_().cuda() 
                self.c0 = weight.new(self.n_layers*self.D, batch_size, self.n_hidden).zero_().cuda() 
                self.hidden = (self.h0, self.c0)
            else:
                self.h0 =  weight.new(self.n_layers*self.D, batch_size, self.n_hidden).zero_() 
                self.c0 = weight.new(self.n_layers*self.D, batch_size, self.n_hidden).zero_() 
                self.hidden = (self.h0, self.c0)
            if self.train_init :
                self.h0 = nn.Parameter(self.h0, requires_grad=True)
                self.c0 = nn.Parameter(self.c0, requires_grad=True)
                self.hidden = (self.h0, self.c0)
        else:
            if (self.train_on_gpu):
                self.h0 = weight.new(self.n_layers*self.D, batch_size, self.n_hidden).zero_().cuda() 
            else:
                self.h0 = weight.new(self.n_layers*self.D, batch_size, self.n_hidden).zero_() 
            if self.train_init :
                self.h0 = nn.Parameter(self.h0, requires_grad=True)
            # print("i got in else, with h: ", type(hidden))
              
        return self.hidden

    def change_batch_size(self, batch_size):
        if batch_size > self.init_batch_size:
            print("we can only work with batch sizes lower than init_batch_size"
                +", increasing it requires re-initalizing and retraining the initial hidden states")
            return

        if batch_size == self.batch_size:
            return

        self.batch_size = batch_size
        #we check if this is the first change of batch size 
        if hasattr(self, 'original_h0'):
            self.h0 = self.original_h0[:, batch_size, :].clone()
            if self.cell == 'lstm' :
                self.c0 = self.original_c0[:, batch_size, :].clone()
        else:
            #clone() preserves the gradients 
            self.original_h0 = torch.clone(self.h0)
            self.h0 = self.original_h0[:, batch_size, :].clone()
            if self.cell == 'lstm' :
                self.original_c0 = torch.clone(self.c0)
                self.c0 = self.original_c0[:, batch_size, :].clone()

    def count_parameters(self):
        table = [["Modules", "shape", "Parameters"]]
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad: continue
            params = parameter.numel()
            shape = list( parameter.shape )
            table.append( [name, shape,  params] )
            total_params+=params
        for line in table :
            print(line)
        print(f"Total Trainable Params: {total_params}")
        return total_params

    def get_accuracy(self, test_data):
        testSRloader = DataLoader(test_data,
                                  batch_size=self.batch_size,
                                  shuffle = True)
        loop =0
        total_accuracy = 0
        for inputs, labels in testSRloader:
            #we ignore last batch if unequal
            if inputs.shape[0] != self.batch_size:
                   break

            accuracy = 0
            if(torch.cuda.is_available()):
                inputs, labels = inputs.cuda(), labels.cuda()
                self = self.cuda()

            if self.cell == 'lstm':
                h = (self.h0, self.c0)
            else:
                h = self.h0

            # get the output from the model
            output, _ = self(inputs, h)   
            for prediction,label in zip(output.squeeze(),labels.squeeze()):
                if torch.argmax(prediction) == torch.argmax(label):
                    accuracy+=1
            total_accuracy+=accuracy/self.batch_size
            loop+=1

        return ( decimal.Decimal(total_accuracy)/decimal.Decimal(loop) ) * 100

    def test_model(self, test_encoded, test_labels):
        print(self)
        global_acc = 0
        
        test_data = TensorDataset(test_encoded, test_labels)
        accuracy = self.get_accuracy(test_data)
        
        print("the accuracy for the test set is: ", str(accuracy), "%")

        # global_acc = global_acc/len(tests)
        # print("global accuracy is: ", global_acc, "%")
        return accuracy

