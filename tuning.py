import training
import torch.nn.functional as F
 

def tune_model_grid(dict, max_len, 
                    train_data, valid_data, 
                    task, nr_classes = None ):
    tune_results =[]
    for lr, hidden_dim, n_layers,cell,bidi, batch_size, patience in training.CartesianProduct(dict["lr_values"], 
                                                                    dict["hidden_dim_values"], 
                                                                    dict["n_layers"],
                                                                    dict["cells"],
                                                                    dict["bidirectional"],
                                                                    dict["batch_size"],
                                                                    dict["patience"]):
        print(lr,hidden_dim,n_layers,cell,bidi)
    
        activation = F.log_softmax
        
        val_loss, net = training.train_model(
                                    train_data,
                                    valid_data,
                                    max_len=max_len, 
                                    hidden_dim=hidden_dim, 
                                    lr=lr, 
                                    n_layers=n_layers,
                                    cell=cell,
                                    bidirectional=bidi,
                                    task=task,
                                    activation = activation,
                                    batch_size=batch_size,
                                    patience=patience,
                                    nr_classes = nr_classes)
        
        net.eval()

        tune_results.append( (val_loss,
                              lr,
                              hidden_dim,
                              n_layers,
                              cell,
                              bidi,
                              net) ) 

    # pick the tuple in tune_results that has the lowest validation loss
    best_model = max(tune_results, key=lambda t:t[0])
    
    print(f"best model with validation loss {best_model[0]}\n"
          f"has learning rate: {best_model[1]}\n"
          f"number of hidden dimensons: {best_model[2]}\n"
          f"number of layers: {best_model[3]}\n"
          f"cell type: {best_model[4]}\n"
          f"bidirectional: {best_model[5]}\n"
          f"neural net: {best_model[6]}")

    return best_model[6]