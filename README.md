# LEARNING FROM RULES GENERALIZING LABELED EXEMPLARS (ICLR 2020) 

This repository provides an implementation of experiments in our [ICLR2020 paper](https://openreview.net/forum?id=SkeuexBtDr)
```
@inproceedings{
Awasthi2020Learning,
title={Learning from Rules Generalizing Labeled Exemplars},
author={Abhijeet Awasthi and Sabyasachi Ghosh and Rasna Goyal and Sunita Sarawagi},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=SkeuexBtDr}
}
```  

# Requirements
This code has been developed with
  - python 3.6
  - tensorflow 1.12.0
  - numpy 1.17.2
  - snorkel 0.9.1
  - tensorflow_hub 0.7.0
  - networkx 2.3.0

# Data Description
We have currently released processed version of 4 datastes used in our paper. 
Following datasets can be found in data/ directory
  * TREC - Question Classification task (Source: http://cogcomp.org/Data/QA/QC/)
  * MITR - Slot filling task (Source: https://groups.csail.mit.edu/sls/downloads/restaurant/)
  * YOUTUBE - Spam Classification task of youtube comments (Source: http://www.dt.fee.unicamp.br/~tiago//youtubespamcollection)
  * SMS - Spam classification task of text messages (Source: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
  * Please see the README files in individual dataset directiories for more information

## data/TREC (or any other data dir) consists following four pickle files
  * d_processed.p (d set: labeled data -- In paper we refer to this is as the "L" dataset) 
  * U_processed.p (U set: unlabeled data -- In paper as well this is referred as "U" dataset)
  * validation_processed.p (validation data)
  * test_processed.p (test data)
  * **NOTE** U_processed.p for YOUTUBE and MITR is unavailable on GitHub due to larger size. You can download entire data dir from [this](https://drive.google.com/file/d/1dWTUC_f0Ks-Hg4TtkHrwWy0RmzLMH-X1/view?usp=sharing) link

## Following objects are dumped inside each pickle file
* x : feature representation of instances
    - shape : [num_instances, num_features]
* l : Class Labels assigned by rules
    - shape : [num_instances, num_rules]
    - class labels belong to {0, 1, 2, .. num_classes-1}
    - l[i][j] provides the class label provided by jth rule on ith instance
    - if jth rule doesn't cover ith instance, then l[i][j] = num_classes (convention)
    - in snorkel, convention is to keep l[i][j] = -1, if jth rule doesn't cover ith instance
* m : Rule coverage mask
    - A binary matrix of shape [num_instances, num_rules]
    - m[i][j] = 1 if jth rule cover ith instance
    - m[i][j] = 0 otherwise
* L : Instance labels
    - shape : [num_instances, 1]
    - L[i] = label of ith instance, if label is available i.e. if instance is from labeled set d
    - Else, L[i] = num_clases if instances comes from the unlabeled set U
    - class labels belong to {0, 1, 2, .. num_classes-1}
* d : binary matrix of shape [num_instances, 1]
    - d[i]=1 if instance belongs to labeled data (d), d[i]=0 otherwise
    - d[i]=1 for all instances is from d_processed.p
    - d[i]=0 for all instances in other 3 pickles {U,validation,test}_processed.p
* r : A binary matrix of shape [num_instances, num_rules]
    - r[i][j]=1 if jth rule was associated with ith instance
    - Highly sparse matrix
    - r is a 0 matrix in all the pickles except d_processed.p
    - Note that this is different from rule coverage mask "m"
    - This matrix defines the coupled rule,example pairs.

# Usage 

From src/hls

* For reproducing numbers in Table 1, Row 1
  - python3 get_rule_related_statistics.py ../../data/TREC 6 None
  - This also provides Majority Vote accuracy in Table2  Column2 (Question dataset) 
* For training, saving and testing a snorkel model  
  - python3 run_snorkel.py ../../data/TREC 6 None  
  - (**RUN THIS BEFORE EXPERIMENTS WHICH DEPEND ON SNORKEL LABELS**) if snorkel model is not already saved in the dataset directory. 
  - We have released pre-trained snorkel models in each dataset directory with name "saved_label_model" )
* For reproducing (approximately) numbers in Table2 Column2 (Question dataset)
  - use train_TREC.sh for training models for different loss functions
  - use test_TREC.sh for testing models for different loss functions
  - best hyperparameters are already set in these scripts
  - both of the above scripts use TREC.sh
* For reproducing numbers (approximately) for other datasets follow steps same as above, with TREC replaced by the dataset name.


# Note:
* f network refes to the classification network
* w network refers to the rule network

# File Description in src/hls
* analyze_w_predictions.py - Used for diagnostics (Old Precision Vs Denoised Precision in Figure 3)
* checkpoint.py - Load/Save checkpoints (Uses code from [checkmate](https://github.com/vonclites/checkmate))
* config.py - All configuration options go here
* data_feeders.py - all kind of data handling for training and testing. 
* data_feeder_utils.py - Load train/test data from processed pickles
* data_utils.py - Other utilities related to data processing
* generalized_cross_entropy_utils.py - Implementation of a [noise tolerant loss functions](https://arxiv.org/pdf/1805.07836.pdf)
* get_rule_related_statistics.py - For reproducing numbers in Table 1
* hls_data_types.py - some basic data types used in data_feeders.py
* hls_model.py - Creates train ops **All the loss functions are defined here**
* hls_test.py - Runs inference using f or w.
  - Inference on f tests the classification network (valid for all the loss functions)
  - Inference on w is used to analyze the denoised rule-precision obtained by w network
  - Inference on w is only meaningful for ImplyLoss and Posterior Reg. method since only these involve a rule (w) network.
* hls_train.py - Two modes:
  - f_d (simply trains f network on labeled data)
  - f_d_U : used for all other modes which utilize unlabeled data
* learn2reweight_utils.py - utilities for implementing [L2R method](https://arxiv.org/abs/1803.09050)
* main.py - entry point
* metrics_utils.py - utilities for computing metrics
* networks.py - implementation of f network (classification network) and w network (rule network)
* pr_utils.py - utilities for implementing Posterior Reg. method 
* run_snorkel.py - training, saving and testing a snorkel model
* snorkel_utils.py - utilitiy to convert l in our format to l in snorkel's format
* test_"DATASET_NAME".sh - model testing (inference) script 
  - e.g. test_TREC.sh runs inference for models trained on TREC dataset
* "train_"DATASET_NAME".sh - model training script
  - e.g. train_TREC.sh trains models on TREC dataset
* "DATASET_NAME".sh - test_"DATASET_NAME".sh and train_"DATASET_NAME".sh use "DATASET_NAME".sh
* utils.py - misc. utilities








