import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

import train_dataset
import fd_testing

def initialize_clients(client_num, model, feature_dim, label_num, device, lr_):
    clients_model_list = {}
    clients_optimizer_list = {}
    for i in range(client_num):
        clients_model_list[i] = model(feature_dim, label_num).to(device)
        clients_optimizer_list[i] = optim.Adam(clients_model_list[i].parameters(), lr=lr_)
    return clients_model_list, clients_optimizer_list


def local_train(model, device, local_dataset, optimizer, local_step = 1):
    model.train()
    for _ in range(local_step):
        data = [torch.from_numpy(sample[0]) for sample in local_dataset]
        target = [sample[1] for sample in local_dataset]
        data = torch.stack(data, dim = 0).to(torch.float32)
        target = torch.LongTensor(target)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


def local_train_proxy_dataset(model, device, proxy_dataset, soft_labels, optimizer, local_step = 500):
    model.train() 
    features = [torch.from_numpy(sample[0]) for sample in proxy_dataset]
    features_tensor = torch.stack(features, dim = 0).to(torch.float32)
    for _ in range(local_step): 
        data = features_tensor.to(device)
        target = soft_labels.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)       
        loss.backward() 
        optimizer.step() 


def global_model(device, epochs, clients_num, clients_data, proxy_dataset, clients_model_list, clients_optimizer_list, test_loader):
    for epoch in range(1, epochs):
        # train local model
        for i in range(clients_num):
            local_train(model=clients_model_list[i], device=device, local_dataset=clients_data[i], optimizer=clients_optimizer_list[i], local_step=1)
        # generate soft labels
        soft_labels = train_dataset.generate_soft_labels(clients_model_list, device, proxy_dataset)
        # transfer global knowledge to each local model
        for i in range(clients_num):
            local_train_proxy_dataset(model=clients_model_list[i], device=device, proxy_dataset=proxy_dataset, soft_labels=soft_labels, optimizer=clients_optimizer_list[i], local_step=1)
        
    test_loss_list = []
    #accuracy_list = []
    Accuracy_list, Precision_list, Recall_list, F1_list = [], [], [], [] 
    for i in range(clients_num):
        test_loss, acc, Accuracy, precision, recall, f1 = fd_testing.test(clients_model_list[i],device, test_loader)
        
        test_loss_list.append(test_loss)
        #accuracy_list.append(acc)
        Accuracy_list.append(Accuracy)
        Precision_list.append(precision)
        Recall_list.append(recall)
        F1_list.append(f1)
            
    Loss = np.mean(test_loss_list)
    #accuracy = np.mean(accuracy_list)
    Accuracy_mean = np.mean(Accuracy_list) 
    Precision_mean = np.mean(Precision_list) 
    Recall_mean = np.mean(Recall_list) 
    F1_mean = np.mean(F1_list) 
    
    print(f'Epoch {epoch}, Loss: {Loss}, F1-score: {F1_mean}')

    return Accuracy_mean, Precision_mean, Recall_mean, F1_mean
    

def local_model(device, epochs, clients_num, clients_data, clients_model_list, clients_optimizer_list, test_loader):
    for epoch in range(1, epochs):
        # train local model
        for i in range(clients_num):
            local_train(model=clients_model_list[i], device=device, local_dataset=clients_data[i], optimizer=clients_optimizer_list[i], local_step=1)
    
    Accuracy_list, Precision_list, Recall_list, F1_list = [], [], [], [] 
    for i in range(clients_num):
        test_loss, acc, Accuracy, precision, recall, f1 = fd_testing.test(clients_model_list[i],device, test_loader)
        
        Accuracy_list.append(Accuracy)
        Precision_list.append(precision)
        Recall_list.append(recall)
        F1_list.append(f1)
            
    Accuracy_mean = np.mean(Accuracy_list) 
    Precision_mean = np.mean(Precision_list) 
    Recall_mean = np.mean(Recall_list) 
    F1_mean = np.mean(F1_list) 
    
    return Accuracy_mean, Precision_mean, Recall_mean, F1_mean
    