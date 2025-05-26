# A Shared Histology-Spatial Transcriptional Data Experimental Platform 🐳

## Authors
Jiaming Liang, Xin Deng

## Requirements
User needs to configure the application environment through the following code:
```
pip install -r requirements/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```
User need to use CLAM package to help split WSI image.
```
git clone https://github.com/mahmoodlab/CLAM.git
```
and pip needed tool package for it.
```
pip install git+https://github.com/oval-group/smooth-topk.git 
```


## Control
All training, verification, visualization and data loading operations are controled by follow file:
```
./config.yml
```
User and follower need to follow certain specifications to modify and add control file content. The specific parameter meanings are recorded in the readme.md appendix.

## Data Loading
Now, this platform support datasets include:
```
TCGA-KRIC (Download version on April 1, 2025)
```


<font color='red'> **Notion** </font>

For all datasets which include histology images, this platform requires **local slicing for the first data loading**, and no online slicing for subsequent loading. Please reserve processed_dir storage space.

Given that the open source data format may be updated over time, this platform is not responsible for the long-term use of the current loading method. User or follower can optimize the platform code in the corresponding data loader and submit Pull Requests, which we will be very grateful for that.

## Appendix
Coming soon...