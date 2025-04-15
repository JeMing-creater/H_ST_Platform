# A Shared Histology-Spatial Transcriptional Data Experimental Platform 🐳

## Authors
Jiaming Liang, Xin Deng

## Requirements
User needs to configure the application environment through the following code:
```
pip install -r requirements/requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

<font color='red'> **Notion** </font>

You may meet some errors when you use **large-image** package. Those errors can be fixed by introduce those code:
```
conda install large-image[all] girder-large-image-annotation[tasks] 
```
You may meet an install error like that: ```Could not find gdal-config. Make sure you have installed the GDAL native library and development headers.```
This time, you need to install gdal first, and need to install by conda. 
```
conda install gdal
```
And then you can continue download packages and install. 

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