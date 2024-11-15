# scKSFD: Federated Distillation Model with Knowledge Sharing for Cell Type Annotation with Clinical Transcriptome Data

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
### Input Data Format
The input data format is shown in the CSV files located in the example_data/ directory:
- expr.csv
- cluster.csv
### Parameters
- clients_num: Client numbers 
- Proportion: Proxy sample proportions

## Contact Us
If you have any questions or suggestions, please contact us via email: [sunn19@tsinghua.org.cn].
