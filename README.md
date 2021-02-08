# GCE embedding layer

## Overview

This repository provides all the necessary code to get reproducible results for our GCE layer. Our GCE layer can be adapted in any context-aware recommender system and allows to outperform the results. It has been tested for different models (MF, FM, NCF, ...) on several datasets, always leading to positive results.

To get all dependencies, run:

    pip install -r requirement.txt

Make sure you have a **CUDA** enviroment to accelarate since these deep-learning models could be based on it. 

We forked the code from [DaisyRec github](https://github.com/AmazingDD/daisyRec), who provide some code for fairly comparison on recommender systems. They handled ranking issue and different types of split in both point-wise and pair-wise problem recommendation. We have extended it to be adapted for context-aware recommendation problems thus focusing in pair-wise problem with time-aware-leave-one-out strategy for splitting and BPR loss as cost function.


## Datasets

You can download experimental data and place it into `data` folder under root project. You can also create `results` folder to store the results. Tensorboard logs will be created under `logs` folder automatically. Pre-processing of data is explained in our paper.

All data are available in links below: 

  - [MovieLens 100K](https://grouplens.org/datasets/movielens/100k/)
  - [User-Book Interactions - full context version Dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/shelves)
  - [LastFM-1K](http://ocelma.net/MusicRecommendationDataset/lastfm-1K.html)
  - [Drug-disease Interactions](https://github.com/luoyunan/DTINet/tree/master/data)
  
  
Other data provided in the code comes from [DaisyRec github](https://github.com/AmazingDD/daisyRec). It is implemented but not tested for this project. We plan to use it in the future.

## Available Baseline Models

| Model | Reference |
|-------|-----------|
| Matrix Factorization | [Baltrunas, 2011). Matrix factorization techniques for context aware recommendation.](https://www.researchgate.net/profile/Bernd-Ludwig/publication/221140971_Matrix_factorization_techniques_for_context_aware_recommendation/links/0deec52b992aa0ec52000000/Matrix-factorization-techniques-for-context-aware-recommendation.pdf) |
| Factorization Machine | [S Rendle, Factorization Machines, 2010.](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) |
| Neural Collaborative Filtering | [HE, Xiangnan, et al. Neural collaborative filtering](https://arxiv.org/abs/1905.08108)
| Neural Factorization Machine | [X He and TS Chua, Neural Factorization Machines for Sparse Predictive Analytics, 2017.](https://arxiv.org/abs/1708.05027) |
| DeepFM | [H Guo, et al. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction, 2017.](https://arxiv.org/abs/1703.04247) |


## Running:

To reproduce the results run:

    * fm-baseline:

	> python main.py --dataset ml-100k --algo_name fm --rankall


    * fm-baseline with GCE: 

	> python main.py --dataset ml-100k --algo_name fm --gce --rankall



## Cite

## Acknowledgements

We refer to the following repositories to improve our code and make it fair in terms of evaluation:

 - Loading, training and evaluation part from [DaisyRec github](https://github.com/AmazingDD/daisyRec)
