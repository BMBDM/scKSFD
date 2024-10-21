import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from preprocessing import preprocess_RNASeq
from scksfd import scksfd

def scKSFD_model(dir, clients_num, Proportion):
    print('Data loading in progress...')
    df = pd.read_csv(dir + 'expr.csv', index_col=0, header=0)
    CellType = pd.read_csv(dir + 'cluster.csv', header=None).values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(CellType)
    print('Data loading complete.')
    print('---------------------------------------------------------------')
    
    print('Data preprocessing in progress...')
    df0 = preprocess_RNASeq(df)
    print('Data preprocessing complete.')
    print('---------------------------------------------------------------')

    print('scKSFD beginning') 
    Accuracy_1, Precision_1, Recall_1, F1_1, run_time_1, Accuracy_2, Precision_2, Recall_2, F1_2, run_time_2 = scksfd(df0.T.values, y, clients_num, Proportion)
    print('scKSFD finished')
    print('---------------------------------------------------------------')

    print('scKSFD results:\n')
    print('Accuracy:', Accuracy_1, '\n')
    print('Precision:', Precision_1, '\n')
    print('Recall:', Recall_1, '\n')
    print('Weighted-F1:', F1_1, '\n')
    print('Run_time:', run_time_1, '\n')

    print('Local model results:\n')
    print('Accuracy:', Accuracy_2, '\n')
    print('Precision:', Precision_2, '\n')
    print('Recall:', Recall_2, '\n')
    print('Weighted-F1:', F1_2, '\n')
    print('Run_time:', run_time_2, '\n')