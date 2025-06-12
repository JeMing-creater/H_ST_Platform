import os
import re
import cv2
import PIL
import time
import math
import monai
import torch
import shutil
import random
import openslide
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from typing import Callable, Optional, Dict, List, Any
from monai.data import Dataset
from functools import lru_cache, reduce
from typing import Iterable, List, Union
from torch.utils.data import Dataset, DataLoader
from monai.transforms import LoadImaged, Resize, Compose, ToTensord, RandFlipd, RandScaleIntensityd, NormalizeIntensityd
from typing import Tuple, List, Mapping, Hashable, Dict

from histolab.slide import Slide
from histolab.util import np_to_pil
from histolab.filters.image_filters import BlueFilter, BluePenFilter, GreenFilter, GreenPenFilter, RedPenFilter
from histolab.tiler import ScoreTiler, GridTiler
from histolab.scorer import NucleiScorer, CellularityScorer
import histolab.filters.image_filters as imf
from histolab.filters.compositions import FiltersComposition
import histolab.filters.morphological_filters as mof
from histolab.filters.image_filters import ImageFilter, Filter, Compose
from histolab.masks import BiggestTissueBoxMask, TissueMask, BinaryMask
from dataset.CLAM import create_patches_fp, extract_features_fp
from dataset.CLAM.dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from dataset.CLAM.models import get_encoder

import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import train_test_split



def check_processed_dir(dir, clean=False):
    if not os.path.exists(dir):
        os.makedirs(dir)
        print(f"{dir} has been created!")
    else:
        if clean==True:
            shutil.rmtree(dir)
            os.makedirs(dir)
            print(f"{dir} exists but has been cleaned!")
        else:
            print(f"{dir} exists!")

def load_tcga_clinical_data(tsv_path):
    # read TSV
    df = pd.read_csv(tsv_path, sep='\t', dtype=str)  # Read all columns as strings to prevent data format issues
    
    # Make sure the case_submitter_id column exists
    if 'cases.submitter_id' not in df.columns:
        raise KeyError("Column 'cases.submitter_id' not found in the TSV file.")
    elif 'demographic.days_to_death' not in df.columns or 'demographic.vital_status' not in df.columns:
        raise KeyError("Column 'demographic.days_to_death' or 'demographic.vital_status' not found in the TSV file.")
    
    # Building a nested dictionary
    clinical_dict = {
        row['cases.submitter_id']: {col: row[col] for col in df.columns if col != 'cases.submitter_id'}
        for _, row in df.iterrows()
    }
    
    return clinical_dict

def find_image_paths(root_dir, need_labels, clinical_dict):
    # Traverse all directories and files under root_dir
    for id_folder in os.listdir(root_dir):
        id_folder_path = os.path.join(root_dir, id_folder)
        if os.path.isdir(id_folder_path):  
            for file in os.listdir(id_folder_path):
                if file.endswith(".svs"):
                    for case_id in clinical_dict.keys():
                        if case_id in file:
                            clinical_dict[case_id]['image_path'] = os.path.join(id_folder_path, file)
                            break
    
    # no file path for this case id.
    cases_to_remove = [case_id for case_id, data in clinical_dict.items() if ('image_path' not in data)]
    cases_to_remove = []
    for case_id, data in clinical_dict.items():
        if 'image_path' not in data:
            print(f"Warning: {case_id} has no file!")
            cases_to_remove.append(case_id)
        elif '--' in data['diagnoses.ajcc_pathologic_m'] and 'm' in need_labels:
            print(f"Warning: {case_id} has no m label!")
            cases_to_remove.append(case_id)
        elif '--' in data['diagnoses.ajcc_pathologic_n'] and 'n' in need_labels:
            print(f"Warning: {case_id} has no n label!")
            cases_to_remove.append(case_id)
        elif '--' in data['diagnoses.ajcc_pathologic_stage'] and 'stage' in need_labels:
            print(f"Warning: {case_id} has no stage label!")
            cases_to_remove.append(case_id)
        elif '--' in data['diagnoses.ajcc_pathologic_t'] and 't' in need_labels:
            print(f"Warning: {case_id} has no t label!")
            cases_to_remove.append(case_id)
        elif '--' in data['demographic.days_to_death'] and 'dd' in need_labels:
            print(f"Warning: {case_id} has no days_to_death label!")
            cases_to_remove.append(case_id)
        elif '--' in data['demographic.vital_status'] and 'vs' in need_labels:
            print(f"Warning: {case_id} has no vital status label!")
            cases_to_remove.append(case_id)
    
    for case_id in cases_to_remove:
        del clinical_dict[case_id]
    
    return clinical_dict

def split_dict_by_ratio(original_dict, ratios=(0.7, 0.1, 0.2), seed=None):
    if seed is not None:
        random.seed(seed)

    keys = list(original_dict.keys())
    random.shuffle(keys)

    total = len(keys)
    num_a = int(total * ratios[0])
    num_b = int(total * ratios[1])
    num_c = total - num_a - num_b  # Make sure the totals match

    keys_a = keys[:num_a]
    keys_b = keys[num_a:num_a+num_b]
    keys_c = keys[num_a+num_b:]

    dict_a = {k: original_dict[k] for k in keys_a}
    dict_b = {k: original_dict[k] for k in keys_b}
    dict_c = {k: original_dict[k] for k in keys_c}

    return dict_a, dict_b, dict_c


def check_scoure_dir(project_config):
    scoure_dir = project_config.clam.processed_path
    if project_config.shutil == True:
        if os.path.exists(scoure_dir):
            shutil.rmtree(scoure_dir)
            print(f"{scoure_dir} has been cleaned!")
        return False
    
    exists = False
    for data_name in ['train', 'val', 'test']:
        data_dir = os.path.join(scoure_dir, data_name)
        if not os.path.exists(data_dir):
            check_processed_dir(data_dir)
        else:
            print(f"{data_dir} exists!")
            exists = True
    return exists


def move_files_to_folder(project_config, datas):
    
    def move_files(project_config, data_dict, data_name='train'):
        # 为满足clam自动化工具所需，构建全WSI文件结构
        target_dir = project_config.clam.processed_path + '/' + data_name
        check_processed_dir(target_dir, clean=True)

        for key in data_dict.keys():
            orl_data_path = data_dict[key]['image_path']
            # 目标文件路径
            file_name = os.path.basename(orl_data_path)
            target_path = os.path.join(target_dir, file_name)
            # 复制文件
            try:
                shutil.copy2(orl_data_path, target_path)
                print(f"successfully copy to : {target_path}")
            except Exception as e:
                print(f"failed to copy because: {e}")

    train_data, val_data, test_data = datas

    move_files(project_config, train_data, data_name='train')
    move_files(project_config, val_data, data_name='val')
    move_files(project_config, test_data, data_name='test')

def print_slide_scale_factors(slides_directory):
    # 确定.svs的最大的放大倍率和下采样倍率
    for filename in os.listdir(slides_directory):
        if filename.endswith(".svs") or filename.endswith(".tiff") or filename.endswith(".ndpi"):  # 根据切片文件的扩展名
            slide_path = os.path.join(slides_directory, filename)
            slide = openslide.OpenSlide(slide_path)
            print(f"Slide: {filename}")
            magnification = slide.properties.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER) 
            print(f"{filename}: Magnification = {magnification}")
            for level in range(slide.level_count):
                downsample = slide.level_downsamples[level]
                print(f" Level {level}: Downsample = {downsample}")

def setting_CLAM_patch(project_config, source, data):
    
    
    save_dir = os.path.join(project_config.clam.processed_path, data + '_' + 'save')
    patch_save_dir = os.path.join(save_dir, 'patches')
    mask_save_dir = os.path.join(save_dir, 'masks')
    stitch_save_dir = os.path.join(save_dir, 'stitches')
    

    directories = {'source': source, 
                'save_dir': save_dir,
                'patch_save_dir': patch_save_dir, 
                'mask_save_dir' : mask_save_dir, 
                'stitch_save_dir': stitch_save_dir} 

    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key not in ['source']:
            os.makedirs(val, exist_ok=True)
        
    seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': True,
				  'keep_ids': 'none', 'exclude_ids': 'none'}
    filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
    vis_params = {'vis_level': -1, 'line_thickness': 250}
    patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}
    parameters = {'seg_params': seg_params,
				  'filter_params': filter_params,
	 			  'patch_params': patch_params,
				  'vis_params': vis_params}
    
    if {
        (not (project_config.clam.patch and os.path.isdir(patch_save_dir) and bool(os.listdir(patch_save_dir)))) 
        or
        (not (project_config.clam.seg and os.path.isdir(mask_save_dir) and bool(os.listdir(mask_save_dir))))
        or
        (not (project_config.clam.stitch and os.path.isdir(stitch_save_dir) and bool(os.listdir(stitch_save_dir))))
    } == True:
        print("Creating needed data...")
        
        # Create patches
        create_patches_fp.seg_and_patch(
            **directories, **parameters,
            patch_size = project_config.clam.patch_size, step_size=project_config.clam.step_size, 
            seg = project_config.clam.seg,  use_default_params=False, save_mask = True, 
            stitch= project_config.clam.stitch,
            patch_level=project_config.clam.patch_level, patch = project_config.clam.patch,
            process_list = None, auto_skip=True
        )

def setting_CLAM_feature(project_config, data):
    save_dir = os.path.join(project_config.clam.processed_path, data + '_' + 'save')
    h5_save_dir = os.path.join(save_dir, 'h5_files')
    pt_save_dir = os.path.join(save_dir, 'pt_files')
    if {
        not (os.path.isdir(h5_save_dir) and bool(os.listdir(h5_save_dir)))
        or
        not (os.path.isdir(pt_save_dir) and bool(os.listdir(pt_save_dir)))
    } == True:
        print('initializing dataset')
        save_dir = os.path.join(project_config.clam.processed_path, data + '_' + 'save')
        data_h5_dir = os.path.join(save_dir, 'patches')
        csv_path = os.path.join(save_dir, 'process_list_autogen.csv')
        data_slide_dir = os.path.join(project_config.clam.processed_path, data)
        bags_dataset = Dataset_All_Bags(csv_path)
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'pt_files'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'h5_files'), exist_ok=True)
        dest_files = os.listdir(os.path.join(save_dir, 'pt_files'))

        model, img_transforms = get_encoder(project_config.clam.model_name, target_img_size=project_config.clam.target_patch_size)

        _ = model.eval()
        model = model.to(project_config.clam.device)
        total = len(bags_dataset)
        loader_kwargs = {'num_workers': 8, 'pin_memory': True} if torch.device(project_config.clam.device).type == "cuda" else {}

        for bag_candidate_idx in range(total):
            slide_id = bags_dataset[bag_candidate_idx].split('.svs')[0]
            bag_name = slide_id+'.h5'
            h5_file_path = os.path.join(data_h5_dir, bag_name)
            slide_file_path = os.path.join(data_slide_dir, slide_id+'.svs')
            print('\nprogress: {}/{}'.format(bag_candidate_idx, total))
            print(slide_id)

            if not project_config.clam.no_auto_skip and slide_id+'.pt' in dest_files:
                print('skipped {}'.format(slide_id))
                continue 
            
            output_path = os.path.join(save_dir, 'h5_files', bag_name)
            time_start = time.time()
            wsi = openslide.open_slide(slide_file_path)
            dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
                                        wsi=wsi, 
                                        img_transforms=img_transforms)

            loader = DataLoader(dataset=dataset, batch_size=project_config.clam.batch_size, **loader_kwargs)

            output_file_path= extract_features_fp.compute_w_loader(output_path, loader = loader, model = model, device=torch.device(project_config.clam.device), verbose = 1)
            
            time_elapsed = time.time() - time_start
            print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

            with h5py.File(output_file_path, "r") as file:
                features = file['features'][:]
                print('features size: ', features.shape)
                print('coordinates size: ', file['coords'].shape)
            
            features = torch.from_numpy(features)
            bag_base, _ = os.path.splitext(bag_name)
            torch.save(features, os.path.join(save_dir, 'pt_files', bag_base+'.pt'))
    else:
        print('h5 files and pt files already exist!')



class PatchH5Dataset(Dataset):
    def __init__(
        self,
        h5_dir: str,
        feature_dir: str,
        clinical_dict: Dict[str, Dict[str, Any]],
        fixed_patch_size: List[int]
    ):
        self.h5_detail = []
        self.clinical_dict = clinical_dict
        self.norm_max, self.norm_min = -1, 99999999
        self.fixed_patch_size = fixed_patch_size

        for fname in os.listdir(feature_dir):
            if not fname.endswith('.h5'):
                continue
            # do not load the h5 file if it does not contain the index in clinical_dict
            for key in clinical_dict.keys():
                if key in fname:
                    if clinical_dict[key]['demographic.days_to_death'] != '--':
                        days_to_death = clinical_dict[key]['demographic.days_to_death'] 
                        if days_to_death == '--':
                            continue
                        else:
                            days_to_death = int(days_to_death)
                        if days_to_death > self.norm_max:
                            self.norm_max = days_to_death
                        if days_to_death < self.norm_min:
                            self.norm_min = days_to_death
                    self.h5_detail.append((
                        os.path.join(feature_dir, fname),
                        clinical_dict[key]['diagnoses.ajcc_pathologic_m'], 
                        clinical_dict[key]['diagnoses.ajcc_pathologic_n'], 
                        clinical_dict[key]['diagnoses.ajcc_pathologic_stage'], 
                        clinical_dict[key]['diagnoses.ajcc_pathologic_t'],
                        clinical_dict[key]['demographic.days_to_death'],
                        clinical_dict[key]['demographic.vital_status']
                                     ))
        
    def encode_label_m(self, label):
        label_map = {'M0': 0, 'M1': 1}
        encoded_label = torch.zeros(len(label_map.keys()))
        if label in label_map:
            encoded_label[label_map[label]] = 1
        return encoded_label

    def encode_label_n(self, label):
        label_map = {'N0': 0, 'N1': 1, 'NX': 2}
        encoded_label = torch.zeros(len(label_map.keys()))
        if label in label_map:
            encoded_label[label_map[label]] = 1
        return encoded_label

    def encode_label_stage(self, label):
        label_map = {'Stage I': 0, 'Stage II': 1, 'Stage III': 2, 'Stage IV': 3}
        encoded_label = torch.zeros(len(label_map.keys()))
        if label in label_map:
            encoded_label[label_map[label]] = 1
        return encoded_label

    def encode_label_t(self, label):
        label_map = {'T1': 0, 'T1a': 1, 'T1b': 2, 'T2': 3, 'T2a': 4, 'T2b': 5, 'T3': 6, 'T3a': 7, 'T3b': 8}
        encoded_label = torch.zeros(len(label_map.keys()))
        if label in label_map:
            encoded_label[label_map[label]] = 1
        return encoded_label
    
    def encode_label_dd(self, label):
        # if label == '--':
        #     return torch.tensor(0).unsqueeze(0)
        # else:
        dd = int(label)
        norm_dd = (dd - self.norm_min) / (self.norm_max - self.norm_min)
        return torch.tensor(norm_dd).unsqueeze(0)
    

    def __len__(self) -> int:
        return len(self.h5_detail)

    def __getitem__(self, idx: int):
        h5_path, m_label, n_label, stage_label, t_label, dd_label, vs_label = self.h5_detail[idx]

        # Load the features from the HDF5 file
        with h5py.File(h5_path, 'r') as f:
            features = torch.tensor(f['features'][:], dtype=torch.float32)
            # coords = torch.tensor(f['coords'][:], dtype=torch.float32)
        
        # Resize the features and coordinates
        features = features.unsqueeze(0).unsqueeze(0)
        features = F.interpolate(features, size=(self.fixed_patch_size[0], self.fixed_patch_size[1]), mode='bilinear', align_corners=False)
        # features = features.squeeze(0).squeeze(0)
        features = features.squeeze(0)


        # with h5py.File(h5_path,'r') as hdf5_file:
        #     features = hdf5_file['features'][:]
        #     coords = hdf5_file['coords'][:]
            
        # Encode label
        m_label_tensor = self.encode_label_m(m_label)
        n_label_tensor = self.encode_label_n(n_label)
        stage_label_tensor = self.encode_label_stage(stage_label)
        t_label_tensor = self.encode_label_t(t_label)
        dd_label = self.encode_label_dd(dd_label)
        vs_label = torch.tensor(1 if vs_label == 'Dead' else 0).unsqueeze(0)

        return {
            'image': features,
            # 'coords': coords,
            'm_label': m_label_tensor,
            'n_label': n_label_tensor,
            'stage_label': stage_label_tensor,
            't_label': t_label_tensor,
            'dd_label': dd_label,
            'vs_label': vs_label
        }

def get_use_data(clinical_dict, project_config, data):
    save_dir = os.path.join(project_config.clam.processed_path, data + '_' + 'save')
    patch_save_dir = os.path.join(save_dir, 'patches')
    feature_save_dir = os.path.join(save_dir, 'pt_files')
    h5_save_dir = os.path.join(save_dir, 'h5_files')
    mask_save_dir = os.path.join(save_dir, 'masks')
    stitch_save_dir = os.path.join(save_dir, 'stitches')
    # TODO: how to use other data?
    dataset = PatchH5Dataset(
        h5_dir=patch_save_dir,
        feature_dir = h5_save_dir,
        clinical_dict=clinical_dict,
        fixed_patch_size=project_config.clam.fixed_patch_size
    )

    return dataset



def get_TCGA_data(config):
    # TODO: introduce more TCGA projects
    if config.trainer.projects == 'KRIC':
        project_config = config.loader.KRIC

    root_dir = project_config["root"]
    # load tsv
    tsv_path = root_dir + '/' + 'clinical.tsv'
    clinical_dict = load_tcga_clinical_data(tsv_path)

    # check if the scoure_dir file exists
    exists = check_scoure_dir(project_config)

    if exists == False:
        # get image path
        clinical_dict = find_image_paths(root_dir, project_config["need_labels"], clinical_dict)
        # split train, val, test data
        train_data, val_data, test_data = split_dict_by_ratio(clinical_dict, 
                                                                ratios=(config.loader.train_ratio, config.loader.val_ratio, config.loader.test_ratio), seed=config.trainer.seed)
        move_files_to_folder(project_config, (train_data, val_data, test_data))
    

    # Clam tools
    for data in ['train','val','test']:
        data_dir = os.path.join(project_config.clam.processed_path, data)
        # ========================= CLAM ==========================
        setting_CLAM_patch(project_config, data_dir, data)
        setting_CLAM_feature(project_config, data)

    # Use h5 file to load data for model training
    # clinical_dict 提供基础图像信息
    # 遍历三个预处理文件夹，获取三个数据集的图像路径
    dataset = {}
    for data in ['train','val','test']:
        dataset[data] = get_use_data(clinical_dict, project_config, data)

    train_loader = monai.data.DataLoader(dataset['train'], num_workers=config.loader.num_workers, batch_size=config.trainer.batch_size, shuffle=True)

    val_loader = monai.data.DataLoader(dataset['val'], num_workers=config.loader.num_workers, batch_size=config.trainer.batch_size, shuffle=False)

    test_loader = monai.data.DataLoader(dataset['test'], num_workers=config.loader.num_workers, batch_size=config.trainer.batch_size, shuffle=True)

    return train_loader, val_loader, test_loader



if __name__ == '__main__':
    import os
    import yaml
    from easydict import EasyDict
    # change to outside directory
    current_path = os.path.dirname(os.path.abspath(__file__))
    parent_path = os.path.dirname(current_path)
    os.chdir(parent_path)

    config = EasyDict(yaml.load(open('config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))

    train_loader, val_loader, test_loader = get_TCGA_data(config)  

    for i, batch in enumerate(train_loader):
        print(batch['image'].shape)
        # print(batch['coords'].shape)
        print(batch['m_label'].shape)
        print(batch['n_label'].shape)
        print(batch['stage_label'].shape)
        print(batch['dd_label'].shape)
        print(batch['dd_label'])
        print(batch['vs_label'].shape)
        print(batch['vs_label'])
