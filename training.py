import class_object
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold  
 

#here we input default values for keyword arguments for testing purposes
def train_model(
                train_data,
                valid_data,  hidden_dim = 4, 
                            max_len = 20,
                            n_layers = 1, 
                            batch_size = 64, 
                            lr = 0.001, 
                            max_epochs = 20, #if early stopping doesn't trigger, we perform max_epochs and then stop
                            cell = 'lstm',
                            bidirectional = False,
                            task = "classif",
                            activation = None,
                            patience = 5,
                            nr_classes = None ):

    
    output_size = nr_classes

    net = class_object.seqRNN(input_size=200, 
                max_len=max_len,
                batch_size=batch_size,
                n_outputs=output_size,
                n_hidden=hidden_dim,
                n_layers= n_layers,
                lr= lr,
                cell=cell,
                bidirectional=bidirectional,
                task=task,
                activation = activation)

    # print(net)

    train_on_gpu = net.train_on_gpu
    # initialize hidden state
    h = net.init_hidden(batch_size)
    

   
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.RMSprop(net.parameters(), lr=lr)


    counter = 0
    print_every = 10

    # move model to GPU, if available
    if(train_on_gpu):
        net.cuda()

    net.train()

    losses =[]
    fold_val_losses = []

    for _ in range(1): 
        fold_val_loss = 0

        trainloader = DataLoader(
                        train_data, 
                        batch_size=batch_size, shuffle = True)
        validloader = DataLoader(
                        valid_data,
                        batch_size=batch_size,shuffle = True)

        trigger = 0
        previous_val_losses = []

        print('--------------------------------')
        # train for some number of epochs
        for e in range(max_epochs):
            if trigger >= patience :
                    print(f"last epoc is {e}")
                    break
            #initialize epoch losses list
            epoch_train_losses = [] 
            
            # batch loop
            for inputs, labels in trainloader:
                
                if trigger >= patience :
                        print(f"we break here")
                        break
                
                # if len(train_data) isn't divisible by batch_size, the last batch will have 
                # len(train_data)%batch_size elements, we can skip this last batch
                if inputs.shape[0] != batch_size:
                   break

                counter += 1
                
                
                if(train_on_gpu):
                    inputs, labels = inputs.cuda(), labels.cuda()

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                
                if net.train_init:
                    if net.cell == 'lstm':
                        h = tuple([net.h0, net.c0])
                    else:
                        h = net.h0

                else: 
                    h = net.init_hidden(batch_size)

                # zero accumulated gradients
                optimizer.zero_grad()

                # get the output from the model
                output, _ = net(inputs, h)

                # if counter == 1:
                    # print(output.shape, labels.shape)
                    # print('output',output,'\n', labels)

                loss = criterion(output.squeeze(), labels.squeeze())


                losses.append(loss.item())
                epoch_train_losses.append(loss.item())
                
                loss.backward()
                
                optimizer.step()

                
                if counter % print_every == 0:
                    
                    # Get validation loss
                    if net.train_init:
                        if net.cell == 'lstm':
                            val_h = tuple([net.h0, net.c0])
                        else:
                            val_h = net.h0
                    else:
                        val_h = net.init_hidden(batch_size)

                        
                    val_losses = []
                    net.eval()
                    for inputs, labels in validloader:

                        #account for case when last element of validloader isn't exactly equal to batch_size
                        if inputs.shape[0] != batch_size:
                            break

                        # Creating new variables for the hidden state, otherwise
                        # we'd backprop through the entire training history
                        # val_h = tuple([each.data for each in val_h])
                        # val_h = net.init_hidden(batch_size)
                        if(train_on_gpu):
                            inputs, labels = inputs.cuda(), labels.cuda()

                        output, _ = net(inputs, val_h)

                        val_loss = criterion(output.squeeze(), labels.squeeze() )
                        # mlflow.log_metric("val losses", val_loss.item(), step = counter )

                        val_losses.append(val_loss.item())

                    if  len(previous_val_losses)!=0 and np.mean(val_losses) > previous_val_losses[-1]:
                            trigger+=1
                            print(f"triggered! value = {trigger} ")
                    elif trigger !=0 :
                        trigger = 0
                        print("trigger reset")

                    previous_val_losses.append(np.mean(val_losses))

                    fold_val_loss = np.mean(val_losses)

                    print("Epoch: {}/{}...".format(e+1, max_epochs),
                        "Step: {}...".format(counter),
                        "Loss: {:.6f}...".format(np.mean(epoch_train_losses)),
                        "Val Loss: {:.6f}".format(np.mean(val_losses)))
                    
                    
                    
                    net.train()

        fold_val_losses.append(fold_val_loss)       
    k_val_loss = np.mean(fold_val_losses)
    print(f'validation loss is: {k_val_loss}' )
    return k_val_loss, net

#code to generate cartesian product of n lists, useful for grid tuning
def CartesianProduct(*args):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    pools = [tuple(pool) for pool in args]
    result = [[]]
    for counter,pool in enumerate(pools):
        # print(f'before: {result} at loop: {counter+1}')
        result = [x+[y] for x in result for y in pool]
        # print(f'after: {result} at loop: {counter+1}')
    for el in result:
        yield tuple(el)
