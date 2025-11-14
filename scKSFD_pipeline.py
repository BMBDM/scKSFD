import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE 
import torch
import time

from preprocessing import preprocess_RNASeq
import train_dataset
import fd_training 
from classify_model import MLPNet, CNNNet, TransformerNet

def scKSFD_model(expr_csv, cluster_csv, clients_num, Proportion):
    ####        Step 0: Data Loaded                  ########################################################
    print('---------------------------------------------------------------')
    print('Step 0: Data loading in progress...')
    df = pd.read_csv(expr_csv, index_col=0, header=0)
    CellType = pd.read_csv(cluster_csv, header=None).values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(CellType)
    print('Step 0: Data loading complete.')  
    
    ####        Step 1: Data Pre-processing          ########################################################
    print('---------------------------------------------------------------')
    print('Step 1: Data preprocessing in progress...')    
    df0 = preprocess_RNASeq(df)
    X = df0.T.values
    print('Step 1: Data preprocessing complete.')

    ####        Step 2: scKSFD Module    ########################################################
    print('---------------------------------------------------------------')
    print('Step 2: scKSFD beginning...') 
    
    torch.manual_seed(10)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)    
    
    print('\t ---------------------------------------------------------')
    print('\t Step 2.1: Creating client data in progress...')
    clients_data = train_dataset.create_clients_data(X_train, y_train, clients_num)    
    print('\t Step 2.1: Creating client data complete.')

    print('\t ---------------------------------------------------------')
    print('\t Step 2.2: Creating proxy dataset in progress...')
    proxy_dataset_size = int(len(y_train) * Proportion)
    # proxy_dataset = train_dataset.create_proxy_dataset_random(clients_data, proxy_dataset_size)
    proxy_dataset = train_dataset.create_proxy_dataset_stratified(clients_data, proxy_dataset_size)
    print('\t Step 2.2: Creating proxy dataset complete.')

    print('\t ---------------------------------------------------------')
    print('\t Step 2.3: Initializing client models...')
    lr = 0.001
    feature_dim = X.shape[1] 
    label_num = np.unique(y).size
    clients_model_list, clients_optimizer_list = fd_training.initialize_clients(clients_num, MLPNet, feature_dim, label_num, device, lr)
    print('\t Step 2.3: Client models initialized.')

    print('\t ---------------------------------------------------------')
    print('\t Step 2.4: Global model running...') 
    epochs = 500
    X_test_tensor = torch.tensor(X_test)
    y_test_tensor = torch.tensor(y_test)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)
    test_kwargs = {'shuffle': True}
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    start_time = time.time()  
    Accuracy_2, Precision_2, Recall_2, F1_2 = fd_training.local_model(device, epochs, clients_num, clients_data, clients_model_list, clients_optimizer_list, test_loader)
    Accuracy_1, Precision_1, Recall_1, F1_1, upload_time, upload_bandwidth, soft_label_bandwidth, download_time, download_bandwidth = fd_training.global_model(device, epochs, clients_num, clients_data, proxy_dataset, clients_model_list, clients_optimizer_list, test_loader)
    
    end_time = time.time()  
    run_time_1 = end_time - start_time  
    print('\t Step 2.4: Global model run finished.')

    print('\t ---------------------------------------------------------')
    print('Step 2: scKSFD finished.')
    print('---------------------------------------------------------------')

    print('scKSFD results:\n')
    print('Accuracy:', Accuracy_1)
    print('Precision:', Precision_1)
    print('Recall:', Recall_1)
    print('Weighted-F1:', F1_1,'\n')

    print('Upload time:', upload_time)
    print('Upload bandwidth:', upload_bandwidth)
    print('Soft label bandwidth:', soft_label_bandwidth)
    print('Download time:', download_time)
    print('Download bandwidth:', download_bandwidth,'\n')

    print('Accuracy_local:', Accuracy_2)
    print('Precision_local:', Precision_2)
    print('Recall_local:', Recall_2)
    print('Weighted-F1_local:', F1_2,'\n')

    print('Run_time:', run_time_1,'\n')
    
    return Accuracy_1, Precision_1, Recall_1, F1_1, upload_time, upload_bandwidth, soft_label_bandwidth, download_time, download_bandwidth, Accuracy_2, Precision_2, Recall_2, F1_2
