# Kernel Methods for Machine Learning : Data Challenge

This repository contains the code, as well as the dataset, used to perform binary classification on molecules using **Weisfeiler-Lehman subtree kernel** and a **SVC**. 

The repository is organized as follow: 
- `data/` folder contains the .pkl files of the training and test data.
- `utils.py` contains all the useful functions used in the project such as the *WL* function and the *KernelSVC* class.
- `dataset_analysis.ipynb` contains the loading, the display and the cleaning of the data as well as a simple statistical overview of the dataset.
- `submission.ipynb` contains the computation of Gram matrices, the SVC fit with stratified cross-validation, and the grid-search on the hyperparameters as well as the final submission.
- `grid_search_results.csv` is a dataframe containing the results from our grid-search. This file is generated by one of the cells in `submission.ipynb`. Each line corresponds to a combination of hyperparameters and contains their values as well as the mean score across the 6 validation sets.
- `report.pdf` is a written summary of our work.


Both authors contributed equally to this project. 