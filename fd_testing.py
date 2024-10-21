import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def test(model, device, test_loader):
    model.eval() 
    test_loss = 0
    correct = 0
    true_labels = [] 
    pred_labels = [] 
    with torch.no_grad(): 
        for data, target in test_loader:
            data = data.to(torch.float32)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)       
            test_loss = test_loss + loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct = correct + pred.eq(target.view_as(pred)).sum().item() 
            true_labels.extend(target.cpu().numpy())
            pred_labels.extend(pred.cpu().numpy().flatten())
    test_loss = np.round(test_loss/len(test_loader.dataset), 4)
    acc = np.round(100. * correct / len(test_loader.dataset), 4)
    Accuracy = np.round(accuracy_score(true_labels, pred_labels), 4)
    precision = np.round(precision_score(true_labels, pred_labels, average='weighted'), 4)
    recall = np.round(recall_score(true_labels, pred_labels, average='weighted'), 4)
    f1 = np.round(f1_score(true_labels, pred_labels, average='weighted'), 4)
    
    return test_loss, acc, Accuracy, precision, recall, f1
