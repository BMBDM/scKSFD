# scKSFD: Federated Distillation Model with Knowledge Sharing for Cell Type Classification with Clinical Transcriptome Data

## Project Overview
scKSFD is a Python toolkit for cell type annotation using clinical transcriptome data, implementing a federated distillation model with knowledge sharing. It aims to facilitate researchers in performing cell type annotation analyses more conveniently.

## Model Architecture
![The Model Architecture of scKSFD](https://github.com/BMBDM/scKSFD/blob/main/scKSFD_model.png)

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

| Gene / Cell | Cell_1 | Cell_2 | Cell_3 | Cell_4 |
|-------------|--------|--------|--------|--------|
| GeneA       |   10   |   0    |   3    |   8    |
| GeneB       |   5    |   2    |   0    |   4    |
| GeneC       |   0    |   7    |   1    |   0    |
| GeneD       |   3    |   3    |   2    |   5    |
| GeneE       |   6    |   1    |   0    |   2    |

cluster.csv example:
| Cell type 1       |
| Cell type 2       |
| Cell type 1       |
| Cell type 3       |


### Parameters
- clients_num: Client numbers 
- Proportion: Proxy sample proportions

### Output
scKSFD results:

Accuracy: 0.9937 
Precision: 0.9938 
Recall: 0.9937 
Weighted-F1: 0.9936 
Run_time: 387.95976758003235 

## Contact Us
If you have any questions or suggestions, please contact us via email: [sunn19@tsinghua.org.cn].
