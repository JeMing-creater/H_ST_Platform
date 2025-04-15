import os
import cv2
import PIL
import time
import math
import yaml
import monai
import torch
import shutil
import random
import numpy as np
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from monai.data import Dataset
from easydict import EasyDict
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


def load_tcga_clinical_data(tsv_path):
    # read TSV
    df = pd.read_csv(tsv_path, sep='\t', dtype=str)  # Read all columns as strings to prevent data format issues
    
    # Make sure the case_submitter_id column exists
    if 'cases.submitter_id' not in df.columns:
        raise KeyError("Column 'cases.submitter_id' not found in the TSV file.")
    
    # Building a nested dictionary
    clinical_dict = {
        row['cases.submitter_id']: {col: row[col] for col in df.columns if col != 'cases.submitter_id'}
        for _, row in df.iterrows()
    }
    
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

def find_image_paths(root_dir, clinical_dict):
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
    cases_to_remove = [case_id for case_id, data in clinical_dict.items() if 'image_path' not in data]
    for case_id in cases_to_remove:
        print(f"Warning: {case_id} has no file!")
        del clinical_dict[case_id]
    
    return clinical_dict

def calculate_target_level(slide, magnification):
    base_mpp_x = float(slide.properties.get('openslide.mpp-x', 1.0))
    desired_mpp = 10.0 / magnification  # Assuming 10 um/px at 1x magnification
    target_level = min(slide.level_count - 1, int(math.log2(base_mpp_x / desired_mpp)))
    return max(0, target_level)

def detect_non_empty_patch(patch):
    # Convert patch to grayscale
    gray = cv2.cvtColor(np.array(patch), cv2.COLOR_RGB2GRAY)
    
    # Threshold the image to binary (black and white)
    _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    
    # Count non-zero pixels (non-empty pixels)
    non_zero_pixels = cv2.countNonZero(binary)
    
    return non_zero_pixels > 0

def encode_label1(label):
    label_map = {'M0': 0, 'M1': 1}
    encoded_label = torch.zeros(len(label_map.keys()))
    if label in label_map:
        encoded_label[label_map[label]] = 1
    return encoded_label

def encode_label2(label):
    label_map = {'N0': 0, 'N1': 1, 'NX': 2}
    encoded_label = torch.zeros(len(label_map.keys()))
    if label in label_map:
        encoded_label[label_map[label]] = 1
    return encoded_label

def encode_label3(label):
    label_map = {'Stage I': 0, 'Stage II': 1, 'Stage III': 2, 'Stage IV': 3}
    encoded_label = torch.zeros(len(label_map.keys()))
    if label in label_map:
        encoded_label[label_map[label]] = 1
    return encoded_label

def encode_label4(label):
    label_map = {'T1': 0, 'T1a': 1, 'T1b': 2, 'T2': 3, 'T2a': 4, 'T2b': 5, 'T3': 6, 'T3a': 7, 'T3b': 8}
    encoded_label = torch.zeros(len(label_map.keys()))
    if label in label_map:
        encoded_label[label_map[label]] = 1
    return encoded_label

def apply_mask_image(img: PIL.Image.Image, mask: np.ndarray) -> PIL.Image.Image:
    img_arr = np.array(img)

    if mask.ndim == 2 and img_arr.ndim != 2:
        n_channels = img_arr.shape[2]
        for channel_i in range(n_channels):
            img_arr[~mask] = [255, 255, 255]  # The position of mask 0 is converted to 255
    else:
        img_arr[~mask] = [255, 255, 255]
    return np_to_pil(img_arr)

class MyMask(BinaryMask):
    def __init__(self, *filters: Iterable[Filter]) -> None:
        self.custom_filters = filters

    @lru_cache(maxsize=100)
    def _mask(self, slide):
        thumb = slide.thumbnail
        if len(self.custom_filters) == 0:
            composition = FiltersComposition(Slide)
        else:
            composition = FiltersComposition(Compose, *self.custom_filters)

        thumb_mask = composition.tissue_mask_filters(thumb)

        return thumb_mask

class BluePenFilter(ImageFilter):
    def __call__(self, img: PIL.Image.Image) -> PIL.Image.Image:
        return self.blue_pen_filter(img)

    def blue_pen_filter(self, img: PIL.Image.Image) -> PIL.Image.Image:
        """Filter out blue pen marks from a diagnostic slide.

        The resulting mask is a composition of green filters with different thresholds
        for the RGB channels.

        Parameters
        ---------
        img : PIL.Image.Image
            Input RGB image

        Returns
        -------
        PIL.Image.Image
            Input image with the blue pen marks filtered out.
        """
        parameters = [
            {"red_thresh": 60, "green_thresh": 120, "blue_thresh": 190},
            {"red_thresh": 120, "green_thresh": 170, "blue_thresh": 200},
            {"red_thresh": 175, "green_thresh": 210, "blue_thresh": 230},
            {"red_thresh": 145, "green_thresh": 180, "blue_thresh": 210},
            {"red_thresh": 37, "green_thresh": 95, "blue_thresh": 160},
            {"red_thresh": 30, "green_thresh": 65, "blue_thresh": 130},
            {"red_thresh": 130, "green_thresh": 155, "blue_thresh": 180},
            {"red_thresh": 40, "green_thresh": 35, "blue_thresh": 85},
            {"red_thresh": 30, "green_thresh": 20, "blue_thresh": 65},
            {"red_thresh": 90, "green_thresh": 90, "blue_thresh": 140},
            {"red_thresh": 60, "green_thresh": 60, "blue_thresh": 120},
            {"red_thresh": 110, "green_thresh": 110, "blue_thresh": 175},
        ]

        blue_pen_filter_img = reduce(
            (lambda x, y: x & y), [self.blue_filter(img, **param) for param in parameters]
        )
        return apply_mask_image(img, blue_pen_filter_img)

    def blue_filter(self,
                    img: PIL.Image.Image, red_thresh: int, green_thresh: int, blue_thresh: int
                    ) -> np.ndarray:
        """Filter out blueish colors in an RGB image.

        Create a mask to filter out blueish colors, where the mask is based on a pixel
        being above a red channel threshold value, above a green channel threshold value,
        and below a blue channel threshold value.

        Parameters
        ----------
        img : PIL.Image.Image
            Input RGB image
        red_thresh : int
            Red channel lower threshold value.
        green_thresh : int
            Green channel lower threshold value.
        blue_thresh : int
            Blue channel upper threshold value.

        Returns
        -------
        np.ndarray
            Boolean NumPy array representing the mask.
        """
        if np.array(img).ndim != 3:
            raise ValueError("Input must be 3D.")
        if not (
                0 <= red_thresh <= 255 and 0 <= green_thresh <= 255 and 0 <= blue_thresh <= 255
        ):
            raise ValueError("RGB Thresholds must be in range [0, 255]")
        img_arr = np.array(img)
        red = img_arr[:, :, 0] > red_thresh
        green = img_arr[:, :, 1] > green_thresh
        blue = img_arr[:, :, 2] < blue_thresh
        return red | green | blue


def get_transforms(config) -> Tuple[
    A.Compose, A.Compose]:
    add_transforms = []
    if config['norm'] != {}:
        add_transforms.append(A.Normalize(mean=config['norm']["mean"], std=config['norm']["std"])) # 归一化
    add_transforms = A.Compose(add_transforms)

    train_transform = A.Compose([
        A.HorizontalFlip(p=config['hf']),# Random vertical flip
        A.VerticalFlip(p=config['vf']),# Random horizontal flip
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=config['rbc']),  # Random intensity scaling
        A.RandomRotate90(p=config['r90']),# Random 90 degree rotation
        add_transforms,
        # ToTensord(keys=["image"])
        ToTensorV2()
    ])
    val_transform   = A.Compose([
        add_transforms,
        # ToTensord(keys=["image"])
        ToTensorV2()
    ])
    return train_transform, val_transform
 

class TCGAImageDataset(Dataset):
    def __init__(self, config, clinical_dict, transforms):
        self.data = []
        self.transforms = transforms
        self.n_tiles = config["n_tiles"]
        self.image_size = config["image_size"]
        self.processed_dir = config["processed_dir"]
        self.check_processed_dir(self.processed_dir)
        self.target_level = config["target_level"]
        for case_id, info in clinical_dict.items():
            if 'image_path' in info and 'diagnoses.ajcc_pathologic_t' in info:
                self.data.append((info['image_path'], info['diagnoses.ajcc_pathologic_m'], info['diagnoses.ajcc_pathologic_n'], info['diagnoses.ajcc_pathologic_stage'], info['diagnoses.ajcc_pathologic_t']))
    
    def check_processed_dir(self, dir, clean=False):
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

    def check_slicepath_and_images(self, slicepath):
        # Check if the base path exists
        if not os.path.exists(slicepath):
            return False
        
        # Construct the scored subdirectory path
        scored_dir = os.path.join(slicepath, "scored")
        
        # Check if the scored subdirectory exists
        if not os.path.isdir(scored_dir):
            return False
        
        # Get all files in the scored subdirectory
        files_in_scored = [f for f in os.listdir(scored_dir) if os.path.isfile(os.path.join(scored_dir, f))]
        
        # Filter out image files (assuming the image extension is .png)
        image_files = [f for f in files_in_scored if f.lower().endswith(('.png'))]
        
        # Check if the number of images is as expected
        if len(image_files) < self.n_tiles:
            return False
    
        return True

    def extract_patches_from_image(self, image_path):
        patch_size   = self.image_size
        target_level = self.target_level
        id = image_path.split('/')[-2]
        processed_path = self.processed_dir + '/' + id + '/'
        self.check_processed_dir(processed_path, True)


        # TODO: Load WSI slices, use use_largeimage mode, but use_largeimage mode can not be used in the current version of histolab.
        tcga_slide = Slide(image_path, processed_path=processed_path, use_largeimage=False, )

        extraction_mask = MyMask()
        extraction_mask.custom_filters = [
            BluePenFilter(),  # Remove blue ink
            imf.RgbToGrayscale(),
            imf.OtsuThreshold(),  # 阈值分割Threshold segmentation
            mof.BinaryErosion(disk_size=1),  # Corrosion
            mof.BinaryDilation(disk_size=1),  # Expansion
            mof.RemoveSmallHoles(area_threshold=500),
            mof.RemoveSmallObjects(min_size=1500), ]

        # Using the Scorer class, estimate the number of cells in H&E stained tiles. This class deconvolutes the hematoxylin channel and uses the portion of the tile occupied by hematoxylin as the cell number fraction. And the check_tissue option helps exclude background areas
        scored_tiles_extractor = ScoreTiler(
            scorer=CellularityScorer(),
            tile_size=patch_size,
            n_tiles=self.n_tiles,
            level=target_level,
            check_tissue=True,
            tissue_percent=80.0,
            pixel_overlap=0,  # default
            prefix="scored/",  # save tiles in the "scored" subdirectory of slide's processed_path
            suffix=".png"  # default
        )

        # Execute Slice
        summary_filename = "summary_ovarian_tiles2.csv"
        SUMMARY_PATH = os.path.join(processed_path, summary_filename)
        locate_img = scored_tiles_extractor.locate_tiles(
            slide=tcga_slide,
            scale_factor=24,  # default
            alpha=128,  # default
            outline="red",  # default
            extraction_mask=extraction_mask
        )
        locate_img.save(os.path.join(processed_path, "locate.png"))

        scored_tiles_extractor.extract(tcga_slide, report_path=SUMMARY_PATH,  extraction_mask=extraction_mask)
    
    def get_slice_image(self, image_path):
        id = image_path.split('/')[-2]
        processed_path = self.processed_dir + '/' + id + '/'

        # Check is the processed path existed or it has enough slice images.
        need_slice = self.check_slicepath_and_images(processed_path)
        if need_slice != True:
            self.extract_patches_from_image(image_path)

        # Load image as tensor
        scored_path = processed_path + '/' + 'scored' + '/'

        files_in_scored = [f for f in os.listdir(scored_path) if os.path.isfile(os.path.join(scored_path, f))]

        image_files = [f for f in files_in_scored if f.lower().endswith(('.png'))]  # filter out image files

        images_tensor_list = []
    
        # Load images and apply transforms
        for img_file in image_files:
            img_path = os.path.join(scored_path, img_file)

            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            augmented_img = self.transforms(image=image)['image']
            images_tensor_list.append(augmented_img)
            
        
        # Stack tensor 
        images_tensor = torch.stack(images_tensor_list)
        
        return images_tensor
            
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        image_path, label1, label2, label3, label4 = self.data[index]
        
        image_tensor = self.get_slice_image(image_path)
        
        label1_tensor = encode_label1(label1)
        label2_tensor = encode_label2(label2)
        label3_tensor = encode_label3(label3)
        label4_tensor = encode_label4(label4)
        return image_tensor, label1_tensor, label2_tensor, label3_tensor, label4_tensor

def get_TCGA_dataloader(config):
    # TODO: introduce more TCGA projects
    for project in config.trainer.projects:
        if project == 'KRIC':
            project_config = config.loader.KRIC
        
        root_dir = project_config["root"]
        # load tsv
        tsv_path = root_dir + '/' + 'clinical.tsv'
        clinical_dict = load_tcga_clinical_data(tsv_path)
        # get image path
        clinical_dict = find_image_paths(root_dir, clinical_dict)
        # split train, val, test data
        train_data, val_data, test_data = split_dict_by_ratio(clinical_dict, 
                                                              ratios=(config.loader.train_ratio, config.loader.val_ratio, config.loader.test_ratio), seed=config.trainer.seed)
        
        # get transforms
        train_transform, val_transform = get_transforms(project_config.transforms)

        train_data = TCGAImageDataset(project_config, train_data, train_transform)
        val_data = TCGAImageDataset(project_config, val_data, val_transform)
        test_data = TCGAImageDataset(project_config, test_data, val_transform)
    
    train_loader = monai.data.DataLoader(train_data, num_workers=config.loader.num_workers,
                                         batch_size=config.trainer.batch_size, shuffle=True)
    val_loader = monai.data.DataLoader(val_data, num_workers=config.loader.num_workers,
                                         batch_size=config.trainer.batch_size, shuffle=False)
    test_loader = monai.data.DataLoader(test_data, num_workers=config.loader.num_workers,   
                                         batch_size=config.trainer.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
    
if __name__ == '__main__':
    config = EasyDict(yaml.load(open('/workspace/Jeming/Pathology/config.yml', 'r', encoding="utf-8"), Loader=yaml.FullLoader))
    
    # 创建数据集
    train_loader, val_loader, test_loader = get_TCGA_dataloader(config)  
    
    start_time = time.time()
    
    for i, batch in enumerate(train_loader):
        print(batch[0].shape)
        print(batch[1].shape)
        print(batch[2].shape)
        print(batch[3].shape)
        print(batch[4].shape)
    
    end_time = time.time()
    elapsed_time_seconds = end_time - start_time  # Calculate the elapsed time in seconds
    elapsed_time_minutes = elapsed_time_seconds / 60  # Convert to minutes
    print(f"Execution Time: {elapsed_time_minutes:.2f} minutes")
    
    