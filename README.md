# Anomaly Detection In Graphs

![CentraleSupelec Logo](https://www.centralesupelec.fr/sites/all/themes/cs_theme/medias/common/images/intro/logo_nouveau.jpg)

This is the repository taken from my end-of-studies yearlong project.

The goal of this project is to build a machine learning model used on a graph representing different users from different organisations and their software usages, in order to **detect anomalies** in the behaviours of said users.

## Author

* Othmane Baziz : othmane.baziz@student-cs.fr

## Requirements

The requirements present in the $requirements.txt$ file and in the $poetry.lock$ file are only working when the CUDA version is CPU. In the case of working with a GPU, we need to install the correct versions and requirements for torch-geometric.

```
!pip install -q torch-scatter -f https://pytorch-geometric.com/whl/torch-1.10.0+${CUDA}.html
!pip install -q torch-sparse -f https://pytorch-geometric.com/whl/torch-1.10.0+${CUDA}.html
!pip install torch-geometric
```
where `${CUDA}` should be replaced by either `cpu`, `cu102`, `cu113` or `cu115` depending on your PyTorch installation (`torch.version.cuda`)

## DOMINANT Algorithm

In the `Dominant.ipynb` notebook, we use everything we have implemented on graphs in order to build an end-to-end encoder/decoder algorithm that will detect anomalies in our nodes. You can find the link of the paper this algorithm is about [here](https://www.public.asu.edu/~kding9/pdf/SDM2019_Deep.pdf).
Due to the large size of our data, I wasn't successful in training my model more than 1 epoch.



