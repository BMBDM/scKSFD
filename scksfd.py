import torch
from sklearn.model_selection import train_test_split
import numpy as np
import time

import train_dataset
import fd_training 
from classify_model import Net

def scksfd(X, y, clients_num, Proportion):
    torch.manual_seed(10)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    
    
    print('Creating client data in progress...')
    clients_data = train_dataset.create_clients_data(X_train, y_train, clients_num)    
    print('Creating client data complete.')

    print('Creating proxy dataset in progress...')
    proxy_dataset_size = int(len(y_train) * Proportion)
    proxy_dataset = train_dataset.create_proxy_dataset(clients_data, proxy_dataset_size)
    print('Creating proxy dataset complete.')
    
    print('Initializing client models...')
    lr = 0.001
    feature_dim = X.shape[1] 
    label_num = np.unique(y).size
    clients_model_list, clients_optimizer_list = fd_training.initialize_clients(clients_num, Net, feature_dim, label_num, device, lr)
    print('Client models initialized.')

    print('Global model running...') 
    epochs = 500
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    test_kwargs = {'shuffle': True}
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    start_time = time.time()  
    Accuracy_1, Precision_1, Recall_1, F1_1 = fd_training.global_model(device, epochs, clients_num, clients_data, proxy_dataset, clients_model_list, clients_optimizer_list, test_loader)
    end_time = time.time()  
    run_time_1 = end_time - start_time  
    print('Global model run finished.')
    
    print('Local model running...')
    start_time = time.time()   
    Accuracy_2, Precision_2, Recall_2, F1_2 = fd_training.local_model(device, epochs, clients_num, clients_data, clients_model_list, clients_optimizer_list, test_loader)
    end_time = time.time()  
    run_time_2 = end_time - start_time  
    print('Local model run finished.')

    return Accuracy_1, Precision_1, Recall_1, F1_1, run_time_1, Accuracy_2, Precision_2, Recall_2, F1_2, run_time_2
     