import os
import cv2
import PIL
import time
import math
import yaml
import h5py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from easydict import EasyDict
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
import anndata as ad
import h5py
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# =======================
# 1. 读取 CytAssist FFPE Human Tonsil 数据
# =======================
def load_cytassist_data(h5_path, spatial_path=None):
    print("Loading expression matrix...")
    with h5py.File(h5_path, 'r') as f:
        data = f['matrix']['data'][:]
        indices = f['matrix']['indices'][:]
        indptr = f['matrix']['indptr'][:]
        shape = f['matrix']['shape'][:]

        barcodes = f['matrix']['barcodes'][:].astype(str)
        features = f['matrix']['features']['name'][:].astype(str)

    # 构建稀疏矩阵
    X = sp.csc_matrix((data, indices, indptr), shape=shape).T.tocsr()

    # 创建 AnnData 对象
    adata = ad.AnnData(X)
    adata.var_names = features
    adata.obs_names = barcodes

    if spatial_path is not None:
        print("Loading spatial coordinates...")
        spatial_df = pd.read_csv(spatial_path, header=None)
        spatial_df.columns = ['barcode', 'in_tissue', 'array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']
        spatial_df = spatial_df.set_index('barcode')

        # 只保留与表达矩阵中重叠的 barcode
        spatial_df = spatial_df.loc[adata.obs_names]

        # 添加空间信息到 obs
        adata.obs[['array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']] = spatial_df[['array_row', 'array_col', 'pxl_row_in_fullres', 'pxl_col_in_fullres']]

    return adata

# =======================
# 2. 数据标准化 + PCA
# =======================
def preprocess_adata(adata, n_pcs=50):
    print("Standardizing and reducing dimensions...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value=10)

    sc.tl.pca(adata, n_comps=n_pcs)
    return adata

# =======================
# 3. 聚类分析
# =======================
def cluster_adata(adata, method='kmeans', n_clusters=10):
    print(f"Clustering using {method}...")
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        adata.obs['cluster'] = kmeans.fit_predict(adata.obsm['X_pca']).astype(str)
    elif method == 'leiden':
        sc.pp.neighbors(adata, n_pcs=30)
        sc.tl.leiden(adata, key_added='cluster')
    else:
        raise ValueError("Unsupported clustering method")
    return adata

# =======================
# 4. 可视化结果
# =======================
def visualize_clusters(adata):
    print("Running UMAP visualization...")
    if 'neighbors' not in adata.uns:
        print("Neighbors not found. Calculating neighbors...")
        sc.pp.neighbors(adata, n_pcs=30)

    sc.tl.umap(adata)
    sc.pl.umap(adata, color='cluster', title='Cluster UMAP')

    if 'pxl_row_in_fullres' in adata.obs.columns:
        print("Plotting spatial view...")
        plt.figure(figsize=(8,8))
        scatter = plt.scatter(
            adata.obs['pxl_col_in_fullres'].astype(float),
            -adata.obs['pxl_row_in_fullres'].astype(float),  # 反向显示
            c=adata.obs['cluster'].astype(int),
            cmap='tab20',
            s=10
        )
        plt.title('Spatial Clustering')
        plt.xlabel('Pixel X')
        plt.ylabel('Pixel Y')
        plt.colorbar(scatter)
        plt.savefig('spatial_clustering.png')
        plt.show()

def get_10x_HT_dataloader(config):
    root = config.loader.tenx_HT.root
    h5_file = root + '/' + 'CytAssist_FFPE_Protein_Expression_Human_Tonsil_filtered_feature_bc_matrix.h5'
    spatial_file = root + '/' + 'spatial' + '/' + 'tissue_positions.csv'

    adata = load_cytassist_data(h5_file, spatial_file)
    adata = preprocess_adata(adata)
    adata = cluster_adata(adata, method='kmeans', n_clusters=8)
    visualize_clusters(adata)


if __name__ == "__main__":
    # Base setting
    config = EasyDict(yaml.load(open('/workspace/Jeming/Pathology/config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    get_10x_HT_dataloader(config)



