import numpy as np
import torch
import random
import pydicom

from torch.utils import data
from torchvision.io import read_image
from monai.data import CacheDataset, DataLoader
from monai.data.utils import worker_init_fn

def create_cacheds_dl(dataset,transforms,cache_rate=None, batch_size=32, shuffle=True, num_workers=0, worker_fn=worker_init_fn, drop_last=False, progress=True, **kwargs):
    '''Create a Monai CacheDataset, a Monai Dataloader. Return the dataloader.'''
    if cache_rate is None:
        cache_rate=0.0
    data_s = CacheDataset(dataset, transforms, cache_rate=cache_rate, progress=progress)
    data_l = DataLoader(data_s, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, worker_init_fn=worker_fn, drop_last=drop_last, **kwargs)
    return data_l

class Dataset_memm(data.Dataset):
    def __init__(self, scan_list, labels, dif_fov=None, shape=(400,512,128), dtype=np.uint8, permute=None, mode="dat", fix_crop=None):
        self.scan_list = scan_list
        self.labels = labels
        self.shape = shape
        self.dtype = dtype
        self.permute = permute
        self.mode = mode
        self.fix_crop = fix_crop
        self.dif_fov = dif_fov
        
    def __len__(self):
        return len(self.scan_list)

    def __getitem__(self, index):

        eyes_paths = self.scan_list[index]
        conv_label = self.labels[index]
        pat_id, visit_id = eyes_paths.split("/")[-3:-1]
        try:
            fov_pos = self.dif_fov[(self.dif_fov['PatientId'] == int(pat_id)) & (self.dif_fov['Visit'] == int(visit_id))]['fovea_pos'].values[0]
        except:
            print("No fovea position found for", pat_id, visit_id)
            fov_pos = self.shape[-1]//2 # Assuming that shape[-1] is the bscan
            
        assert fov_pos,'No fovea position found!'

        #28.10.2024
        if fov_pos <self.shape[-1]//2 - 10 or fov_pos > self.shape[-1]//2 + 10:
            #print("Fovea position is not in the middle!", fov_pos)
            fov_pos = self.shape[-1]//2

        if self.mode == "dat":
            eyes = np.array(np.memmap(eyes_paths, dtype=self.dtype, mode='r', shape=self.shape))
        elif self.mode == "npz":
            eyes = np.load(eyes_paths)['img']
        elif self.mode == "npy":
            eyes = np.load(eyes_paths)
        elif self.mode == "dcm":
            eyes = np.array(pydicom.dcmread(eyes_paths).pixel_array)
            eyes = np.array(eyes)
        elif self.mode == "png":
            eyes = read_image(eyes_paths)
            eyes = np.array(eyes)

        flag = (not np.any(eyes)) or np.any(np.isnan(eyes))
        if flag:
            print("empty array!", eyes_paths)
        
        if str(eyes.dtype) != self.dtype:
            eyes = eyes.astype(self.dtype)

        if self.fix_crop:
            eyes = eyes[self.fix_crop]

        if self.permute:
            eyes = np.transpose(eyes, self.permute)


        return_dict = {"image": eyes, "label": conv_label, "path": eyes_paths,'patID': pat_id, 'visitID': visit_id, 'fovea_pos': fov_pos}

        return return_dict
    
class TemporalDataset(data.Dataset):
    def __init__(self, dataset, transforms, cache_rate = 0.0, min_max=(90,540), sorted=False):
        self.scan_list = dataset.scan_list
        self.sorted = sorted
        # Since this is dictionary there is already just one key patient, not multiple
        self.pat_keys = {key.split('/')[-3] : [value.split('/')[-2] for value in self.scan_list if value.split('/')[-3]==key.split('/')[-3]]
                        for key in self.scan_list}
        
        # Remove patients with less than 15 visits
        self.pat_keys = {key: value for key, value in self.pat_keys.items() if len(value) > 15}
        dataset.scan_list = [x for x in self.scan_list if x.split('/')[-3] in list(self.pat_keys.keys())] # Update the scanlist of base dataset
        self.scan_list = dataset.scan_list # update back self list as well

        self.patient_list = list(self.pat_keys.keys())
        self.min_max = min_max

        self.cached_dataset = CacheDataset(dataset, transforms, cache_rate=cache_rate)
        
    def __len__(self):
        return len(self.pat_keys.keys())

    def __getitem__(self, index):
        patient = self.patient_list[index]

        seed = random.randint(0,2**32)
        random.seed(seed)
        torch.manual_seed(seed)

        eye1, eye2 = None, None

        while not eye1 or not eye2:
            # If minimum is 0, then we can have same dates. Else no repetitio
            dates = random.choices(self.pat_keys[patient], k=2) if self.min_max[0] == 0 else random.sample(self.pat_keys[patient], k=2) 
            #TODO think something better for the case where one of the date is 7
            dates_temp = [0 if int(x)==7 else x for x in dates]
            if self.sorted: dates = sorted(dates) 
            diff = int(dates_temp[1]) - int(dates_temp[0])
            if abs(diff) <= self.min_max[1] and abs(diff) >= self.min_max[0]:
                #get the positions for the cached dataset
                temp_pos1 = [i for i, s in enumerate(self.scan_list) if (str(dates[0]).zfill(3) in s) and (str(patient) in s)][0]
                temp_pos2 = [i for i, s in enumerate(self.scan_list) if (str(dates[1]).zfill(3) in s) and (str(patient) in s)][0]

                eye1 = self.cached_dataset[temp_pos1]
                seed = random.randint(0,2**32)
                random.seed(seed)
                torch.manual_seed(seed)
                eye2 = self.cached_dataset[temp_pos2]
            
        time_diff = np.float32((abs(diff)-self.min_max[0])/(self.min_max[1]-self.min_max[0]))
        return_dict = {"image": eye1["image"], "image_1":eye2["image"], "label": time_diff, 'patID': patient, 'visitID': eye1['visitID'], 'visitID_1': eye2['visitID']}#, 'fovea_pos': eye2['fovea_pos']}#, "loop":i}#, "dates": dates, "paths": [self.scan_list[temp_pos1], self.scan_list[temp_pos2]], 'paths_cac': [self.cached_dataset.data.scan_list[temp_pos1], self.cached_dataset.data.scan_list[temp_pos2]]}

        return return_dict