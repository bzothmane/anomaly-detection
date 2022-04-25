# Anomaly Detection In Graphs

## Download Data

On your node gpu, please excute these commands to download datasets.

wget https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop/download

Once you have downloaded them, use these commands to unzip. This may take some time, please wait with your patience.

tar -xzf archive.tar.gz

## Preprocessing 

We excute the notebook Data_Base.ipynb to get a file named data_out.csv : We merged some incomplete data set and added columns to have the desired table.

## Fake Data Generation

In order to simulate fraudulous activity in the costumer base. We excute the code Data_Generation.py.

## Graph Generation

- On the file Graph_Generation.py : 
    - Put the name of the name of the data file in database_path.
    - Put the name of the fake data file in fake.
    - Put the path of the repositiry where we found the checkpoint file in path.
    
- On the file batch.sh 
    - we put the command python3 graph_Generation.py

- we excute the command : bash run.sh sbatch

## Embedding 

we excute the file metapath2vec.py

## Anomaly Detection 

In order to test the clustering methods on a part of the dataset we can use the notebook Anomaly_Detection_Notebook.ipynb





