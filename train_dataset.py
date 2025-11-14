import numpy as np
import random
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedShuffleSplit


def select_representative_sample_per_label(X,y):
    X_samples,y_samples = [],[]
    y_set = set(list(y))
    for y_label in y_set:
        for x_sample,y_sample in zip(X,y):
            if y_sample == y_label:
                X_samples.append(x_sample)
                y_samples.append(y_sample)
                break 
    return X_samples,y_samples


def create_clients_data(X_train, y_train, clients_num):
    clients_data = [None] * clients_num
    data = list(zip(X_train, y_train))
    random.shuffle(data)
    size = len(data) // clients_num
    shards = [data[i:i + size] for i in range(0, size * clients_num, size)]
    X_samples, y_samples = select_representative_sample_per_label(X_train,y_train)
    new_shards = []
    adddata = list(zip(X_samples, y_samples))
    for shard in shards:
        shard.extend(adddata)
        new_shards.append(shard) 
    assert (len(new_shards) == clients_num)
    clients_data = [new_shards[i] for i in range(clients_num)]
    return clients_data


def random_sampling(data, size):
    sampled_data = []
    num = data.__len__()
    random_index = np.random.choice(num, size = size, replace = False)
    for i in range(len(random_index)):
        random_data = data.__getitem__(random_index[i])
        sampled_data.append(random_data)
    return sampled_data 
 
def create_proxy_dataset_random(clients_data, proxy_dataset_size):
    proxy_dataset = []
    for data in clients_data:
        size = proxy_dataset_size // len(clients_data)
        data_sampling = random_sampling(data, size)
        proxy_dataset.extend(data_sampling)    
    np.random.shuffle(proxy_dataset)
    return proxy_dataset


def stratified_sampling(data, size):
    X = [x[0] for x in data]
    y = [x[1] for x in data]
    sampled_data = []
    stratified_sampler = StratifiedShuffleSplit(n_splits=1, test_size=size, random_state=42)    
    for _, test_index in stratified_sampler.split(X, y):
        X_sampled = [X[i] for i in test_index]
        y_sampled = [y[i] for i in test_index]
    sampled_data = list(zip(X_sampled, y_sampled))    
    return sampled_data
    
def create_proxy_dataset_stratified(clients_data, proxy_dataset_size):
    proxy_dataset = []
    for data in clients_data:
        size = proxy_dataset_size // len(clients_data)
        data_sampling = stratified_sampling(data, size)        
        proxy_dataset.extend(data_sampling)  
    np.random.shuffle(proxy_dataset)  
    return proxy_dataset

def create_proxy_dataset_stratified_adaptive(clients_data, proxy_dataset_size, clients_num):
    proxy_dataset = []
    for data in clients_data:
        adaptive_size = proxy_dataset_size // (clients_num if clients_num <= 10 else 10) 
        data_sampling = stratified_sampling(data, adaptive_size)        
        proxy_dataset.extend(data_sampling)  
    np.random.shuffle(proxy_dataset)
    return proxy_dataset
    
def generate_soft_labels(model_list, device, proxy_dataset):
    features = [torch.from_numpy(sample[0]) for sample in proxy_dataset]
    labels = [sample[1] for sample in proxy_dataset]
    features_tensor = torch.stack(features, dim = 0).to(torch.float32)
    labels_tensor = torch.LongTensor(labels)   
    with torch.no_grad(): 
        soft_labels = 0.0
        for i in range(len(model_list)):
            model_list[i].eval()
            features_tensor = features_tensor.to(device)
            output = model_list[i](features_tensor)
            softlabel = F.softmax(output, dim = 1)
            soft_labels += softlabel * torch.reshape(labels_tensor[i], (-1,1) ).to(device)   
    return soft_labels

def generate_hard_labels(model_list, device, proxy_dataset):
    features = [torch.from_numpy(sample[0]) for sample in proxy_dataset]  
    labels = [sample[1] for sample in proxy_dataset] 
    features_tensor = torch.stack(features, dim = 0).to(torch.float32)
    labels_tensor = torch.LongTensor(labels)
    with torch.no_grad(): 
        model_outputs = []  
        for i in range(len(model_list)):
            model_list[i].eval()
            output = model_list[i](features_tensor.to(device))
            model_outputs.append(output)  
        averaged_output = torch.mean(torch.stack(model_outputs, dim=0), dim=0)  
        hard_labels = torch.argmax(averaged_output, dim=1)  
    return hard_labels