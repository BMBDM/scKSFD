# scKSFD: Federated Distillation Model with Knowledge Sharing for Cell Type Classification of Clinical Transcriptome Data

## Project Overview
scKSFD is a Python toolkit for cell type classification using clinical transcriptome data, implementing a federated distillation model with knowledge sharing. 

## Model Architecture
![The Model Architecture of scKSFD](https://github.com/BMBDM/scKSFD/blob/main/scKSFD_model.png)

The model is realized through four steps: 
* Clients independently train local models ${f_1, f_2, ..., f_K}$; 
* The local model generates the soft labels of proxy dataset and sends them to the server $p_k^((i) )=softmax(f_k ((x_i ) ̂ )), k=1,2,...,K, i=1,2,...,n$; 
* The server aggregates prediction results and sends them to the clients $(y_i ) ̂  =∑_(k=1)^K〖1_[y_i=k] ⋅p_k^((i) ) 〗,i=1,2,...,n$; 
* Clients utilize soft labels for knowledge distillation. 

## Installation Guide
- Clone the repository:
  ```
  git clone https://github.com/BMBDM/scKSFD.git
  cd scKSFD
  ```
- Install the dependencies:
  ```
  pip install -r requirements.txt
  ```
- Optionally, install the toolkit as an importable Python package:
  ```
  pip install .
  ```
  
## Usage Example
  ```
  from main import scKSFD_model
  clients_num = 2
  Proportion = 0.1
  scKSFD_model('example_data/', clients_num, Proportion)
  ```
### Input
The input data format is shown in the CSV files located in the example_data/ directory:
- expr.csv
- cluster.csv

expr.csv example:

|             | Cell_1 | Cell_2 | Cell_3 | Cell_4 |
|-------------|--------|--------|--------|--------|
| GeneA       |   10   |   0    |   3    |   8    |
| GeneB       |   5    |   2    |   0    |   4    |
| GeneC       |   0    |   7    |   1    |   0    |
| GeneD       |   3    |   3    |   2    |   5    |
| GeneE       |   6    |   1    |   0    |   2    |

cluster.csv example:

| Cell type 1       |
|-------------      |
| Cell type 2       |
| Cell type 1       |
| Cell type 3       |


### Parameters
- clients_num: Client numbers, e.g. 2
- Proportion: Proxy sample proportions, e.g. 0.1

### Output
  ```
  scKSFD results:       	
  Accuracy: 0.9937        	
  Precision: 0.9938        	
  Recall: 0.9937       		
  Weighted-F1: 0.9936        	
  Run_time: 387.95976758003235 
  ``` 

## Contact Us
If you have any questions or suggestions, please contact us via email: [sunn19@tsinghua.org.cn].
