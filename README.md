# Anomaly Detection In Graphs

![CentraleSupelec Logo](https://www.centralesupelec.fr/sites/all/themes/cs_theme/medias/common/images/intro/logo_nouveau.jpg)


## Authors 
* Othmane Baziz : othmane.baziz@student-cs.fr
* Oualid Essaid : oualid.essaid@student-cs.fr
* Asma Mairech : asma.mairech@student-cs.fr

## Introduction

This is the repository from our end-of-study project on anomaly detection in graphs. 

## Requirements

python==3.9.5
pytorch==1.10

```
!pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.10.0+${CUDA}.html
!pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.10.0+${CUDA}.html
!pip install torch-geometric
```
where `${CUDA}` should be replaced by either `cpu`, `cu102`, `cu113` or `cu115` depending on your PyTorch installation (`torch.version.cuda`)

## Download Data

On your node gpu, please execute these commands to download the datasets we used throughout our project.

`wget https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop/download`

Once you have downloaded them, use these commands to unzip. This may take some time, please arm yourself with patience. :smile:

`tar -xzf archive.tar.gz`

## Preprocessing

In order to be able to use our data, we will transform it into one big .csv file. Furthermore, in order to test our results, we will generate fake data that resembles fraudulous activity, so that we can detect it.

### Creating our datasets

We merged all our incomplete datasets and added some columns we needed in order to have our tabular dataset that we will use through out this project, namely `data_out.csv`. 
In order to generate it, it is needed to execute the `Data_Base.ipynb` file.

### Fake Data Generation

In order to simulate fraudulous activity in the customer base. We execute the code `Data_Generation.py` in order to generate a `fake_data.csv` file that we'll use to test our algorithms.

## Graph Generation

- On the file Graph_Generation.py : 
    - Put the name of the name of the data file in database_path.
    - Put the name of the fake data file in fake.
    - Put the path of the repositiry where we found the checkpoint file in path.
    
- On the file batch.sh 
    - we put the command python3 graph_Generation.py

- we excute the command : bash run.sh sbatch

## Embedding 

We execute the file metapath2vec.py

## Anomaly Detection 

In order to test the clustering methods on a part of the dataset we can use the notebook Anomaly_Detection_Notebook.ipynb

## DOMINANT Algorithm

In the `Dominant.ipynb` notebook, we use everything we have implemented on graphs in order to build an end-to-end encoder/decoder algorithm that will detect anomalies in our nodes. You can find the link of the paper this algorithm is about [here](https://www.public.asu.edu/~kding9/pdf/SDM2019_Deep.pdf).
Due to the large size of our data, I wasn't successful in training my model more than 1 epoch.



