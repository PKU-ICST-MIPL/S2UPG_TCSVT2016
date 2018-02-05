# Introduction
This is the source code of our TCSVT 2016 paper "Semi-Supervised Cross-Media Feature Learning with Unified Patch Graph Regularization", Please cite the following paper if you use our code.

Yuxin Peng, Xiaohua Zhai, Yunzhen Zhao, and Xin Huang, "Semi-Supervised Cross-Media Feature Learning with Unified Patch Graph Regularization", IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), Vol. 26, No. 3, pp. 583-596 , Mar. 2016. [[PDF]](http://www.icst.pku.edu.cn/mipl/tiki-download_file.php?fileId=256)

# Usage
Run our script to train and test:
 
    S2UPG.m

The parameters are as follows:

    I_tr_NP: the feature matrix of image instances for training, dimension : tr_n * d_i
    T_tr_NP: the feature matrix of text instances for training, dimension : tr_n * d_t
    I_te_NP: the feature matrix of image instances for test, dimension : te_n * d_i
    T_te_NP: the feature matrix of text instances for test, dimension : te_n * d_t
    I_tr_P: the feature matrix of image patches for training, dimension : tr_n * d_i
    T_tr_P: the feature matrix of text patches for training, dimension : tr_n * d_t
    I_te_P: the feature matrix of image patches for test, dimension : te_n * d_i
    T_te_P: the feature matrix of text patches for test, dimension : te_n * d_t
    trainCat: the category list of data for training, dimension : tr_n * 1
    testCat: the category list of data for test, dimension : te_n * 1
    gamma: sparse regularization parameter, default: 1000
    sigma: mapping regularization parameter, default: 0.1
    miu: high level regularization parameter, default: 10
    k: kNN parameter, default: 100

The source codes are for Wikipedia dataset, which can be download via: http://www.svcl.ucsd.edu/projects/crossmodal/.

For more information, please refer to our [paper](http://www.icst.pku.edu.cn/mipl/tiki-download_file.php?fileId=256)

# Our Related work
If you are interested in cross-media retrieval, you can check our recently published overview paper on IEEE TCSVT:

Yuxin Peng, Xin Huang, and Yunzhen Zhao, "An Overview of Cross-media Retrieval: Concepts, Methodologies, Benchmarks and Challenges", IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), 2017.[[PDF]](http://www.icst.pku.edu.cn/mipl/tiki-download_file.php?fileId=376)

Welcome to our [Benchmark Website](http://www.icst.pku.edu.cn/mipl/xmedia) and [Laboratory Homepage](http://www.icst.pku.edu.cn/mipl) for more information about our papers, source codes, and datasets.
