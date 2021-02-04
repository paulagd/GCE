#GCE embedding layer

## Overview

This repository provides all the necessary code to get reproducible results for our GCE layer. Our GCE layer can be adapted in any context-aware recommender system and allows to outperform the results. It has been tested for different models (MF, FM, NCF, ...) on several datasets, always leading to positive results.

To get all dependencies, run:

    pip install -r requirement.txt

Make sure you have a **CUDA** enviroment to accelarate since these deep-learning models could be based on it. 

We forked the code from [DaisyRec github](https://github.com/AmazingDD/daisyRec) which handled ranking issue mainly and split recommendation problem into point-wise ones and pair-wise ones so that different loss function are constructed such as BPR, Top-1, Hinge and Cross Entropy. We adapted their code to context-aware recommendation focusing in pair-wise problem and BPR loss. 

## Datasets

You can download experiment data, and put them into the `data` folder. You can also create `results` folder to store the results. Tensorboard logs will be created under `logs` folder automatically.

All data are available in links below: 

  - [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
  - [User-Book Interactions - full context version Dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/shelves)
  - [LastFM-1K](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html)
  - [Drug-disease Interactions](https://github.com/luoyunan/DTINet/tree/master/data)
  
  
Other data provided in the code comes from [DaisyRec github](https://github.com/AmazingDD/daisyRec). It is implemented but not tested for this project. We plan to use it in the future.

## Available Baseline Models

| Model | Reference |
|-------|-----------|
| Factorization Machine | [S Rendle, Factorization Machines, 2010.](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) |
| Neural Collaborative Filtering | [HE, Xiangnan, et al. Neural collaborative filtering](https://arxiv.org/abs/1905.08108)
| Wide&Deep | [HT Cheng, et al. Wide & Deep Learning for Recommender Systems, 2016.](https://arxiv.org/abs/1606.07792) |
| Neural Factorization Machine | [X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.](https://arxiv.org/abs/1708.05027) |
| DeepFM | [H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.](https://arxiv.org/abs/1703.04247) |


## Cite

## Acknowledgements

We refer to the following repositories to improve our code and make it fair in terms of evaluation:

 - Loading, training and evaluation part from [DaisyRec github](https://github.com/AmazingDD/daisyRec)
