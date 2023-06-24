
# coding: utf-8



from __future__ import division
from __future__ import print_function

# In[110]:

import numpy as np
import pandas as pd
import pydicom
import os
import matplotlib.pyplot as plt
import collections
# from tqdm import tqdm_notebook as tqdm
from datetime import datetime

from math import ceil, floor
import cv2
import sys
# from sklearn.model_selection import ShuffleSplit

def window_image(img, window_center,window_width, intercept, slope):
    
#     window_center,window_width = 50 ,100
    img = (img*slope +intercept)
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    return img 


def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def _normalize(img):
    if img.max() == img.min():
        return np.zeros(img.shape)-1
    return 2 * (img - img.min())/(img.max() - img.min()) - 1

def normalize_minmax(img):
    mi, ma = img.min(), img.max()
    if mi == ma:
        return np.zeros(img.shape)-1
    return 2*(img - mi) / (ma - mi) - 1

def getName(s):
    ix1 = s.rfind('/')
    ix2 = s.rfind('.')
    return s[ix1:ix2]


def _read(path, desired_size = (512,512)):
    """Will be used in DataGenerator"""

    try:
        data = pydicom.read_file(path)
        image = data.pixel_array
        window_center , window_width, intercept, slope = get_windowing(data)
        
        image_windowed = window_image(image, window_center, window_width, intercept, slope)
        img = normalize_minmax(image_windowed)

    except:
        img = np.zeros(desired_size[:2])-1
    
    if img.shape[:2] != desired_size[:2]:
        print("image shape is not desired size. Interpolation is done.")
        img = cv2.resize(img, desired_size[:2], interpolation=cv2.INTER_LINEAR)
    
    
    return img



import os
import numpy as np
import pydicom

D_dir2header_df = {}


def get_dicom_header_df(image_dir , labels = []):
    global D_dir2header_df
    if image_dir in D_dir2header_df:
        return D_dir2header_df[image_dir]

    # image_dir = row['Image File Path']


    labels = ['ImageName','InstanceNumber',
            'BitsAllocated', 'BitsStored', 'Columns', 'HighBit', 
            'ImageOrientationPatient_0', 'ImageOrientationPatient_1', 'ImageOrientationPatient_2',
            'ImageOrientationPatient_3', 'ImageOrientationPatient_4', 'ImageOrientationPatient_5',
            'ImagePositionPatient_0', 'ImagePositionPatient_1', 'ImagePositionPatient_2',
            'Modality', 'PatientID', 'PhotometricInterpretation', 'PixelRepresentation',
            'PixelSpacing_0', 'PixelSpacing_1', 'RescaleIntercept', 'RescaleSlope', 'Rows', 'SOPInstanceUID',
            'SamplesPerPixel', 'SeriesInstanceUID', 'StudyID', 'StudyInstanceUID', 
            'WindowCenter', 'WindowWidth', 
        ] if not labels else labels

    data = {l: [] for l in labels}
    
    ctList = os.listdir(image_dir)
    ctList.sort()

    for image in ctList:
        if '.dcm' not in image:
            continue
        if os.path.getsize(os.path.join(image_dir, image)) < 5*1024:
            print('%s size < 5kb skiped!'%os.path.join(image_dir, image) )
            continue
        data["ImageName"].append(image)

        ds = pydicom.dcmread(os.path.join(image_dir, image))
        for metadata in ds.dir():
            if metadata not in data and metadata not in ['ImageOrientationPatient','ImagePositionPatient','PixelSpacing']:
                continue
            if metadata != "PixelData":
                metadata_values = getattr(ds, metadata)
                if type(metadata_values) == pydicom.multival.MultiValue and metadata not in ["WindowCenter", "WindowWidth"]:
                    for i, v in enumerate(metadata_values):
                        data[f"{metadata}_{i}"].append(v)  
                else:
                    if type(metadata_values) == pydicom.multival.MultiValue and metadata in ["WindowCenter", "WindowWidth"]:
                        data[metadata].append(metadata_values[0])
                    else:
                        data[metadata].append(metadata_values)

    df_image = pd.DataFrame(data).set_index("InstanceNumber")
    D_dir2header_df[image_dir] = df_image
    return df_image


# In[2]:

def InstanceNumber2file_name(df_image, num):
    return df_image.loc[num,'ImageName']

def InstanceNumber2data_element(df_image, num, label):
    return df_image.loc[num , label]

    
def get_SliceThickness(df_image):
    flag = False
    L = df_image['ImagePositionPatient_2'].tolist()
    thick = list( np.diff(L) )
    res = float( max(set(thick), key=thick.count) )
    res = -res if res < 0 else res
    
    L.sort()
    thick2 = list( np.diff(L) )
    res2 = float( max(set(thick2), key=thick2.count) )
    if res2 ==0 and res==0:
        result = 0
        flag = True
        print('Warning intv is 0')
        print(df_image['ImagePositionPatient_2'])
    if res2 == res:
        result = res
    else:
        result = res
        flag = True
        print('Warning intv may wrong',res,res2)
        print(df_image['ImagePositionPatient_2'])
    
    return result 

def InstanceNumber2windows_min_max(df_image,num):
    try:     
        WL = InstanceNumber2data_element(df_image, num, 'WindowCenter')
        WW = InstanceNumber2data_element(df_image, num, 'WindowWidth')
    except:
        print("Warning! Window Center or Width is empty! Now use default values")
        WL , WW = 250 , 1500
        
    minHU = int( WL-WW/2 )
    maxHU = minHU + int(WW)
    return [minHU , maxHU]


class ASerial:
    P=-1
    D=-1
    S=-1
    name = ''
    def __init__(self, path_str):
        self.path = path_str
        self.getP()
        self.getD()
        self.getS()
        self.convert_path()
        
    def getP(self, target = 'DeepLesion_', L=6):
        ix = self.path.rfind(target) + len(target)
        ss = self.path[ix:ix+L]
        self.P = int(ss)
        
    def getD(self, target = '/D', L=6):
        ix = self.path.rfind(target) + len(target)
        ss = self.path[ix:ix+L]
        self.D = int(ss)
        
    def getS(self, target = '/S', L=6):
        ix = self.path.rfind(target) + len(target)
        ss = self.path[ix:ix+L]
        self.S = int(ss)
        
    def convert_path(self):
        self.name = '%06d_%02d_%02d'%(self.P, self.D, self.S)




import os
import cv2
import json, yaml
import numpy as np
from PIL import Image
# from collections import OrderedDict
from pycocotools import mask as cocomask
from pycocotools import coco as cocoapi



def replacer(s, newstring, index, nofail=False):
    # raise an error if index is outside of the string
    if not nofail and index not in range(len(s)):
        raise ValueError("index outside given string")

    # if not erroring, but the index is still not in the correct range..
    if index < 0:  # add it to the beginning
        return newstring + s
    if index > len(s):  # add it to the end
        return s + newstring

    # insert the new string between "slices" of the original
    return s[:index] + newstring + s[index + 1:]

def convert_file_name(name,S='/'):
    ix = name.rfind('_')
    return replacer(name,S,ix)

def file_name2id(name):
    name.replace('.png','')
    name.replace('_','')
    return int('1' + name)
    
def get_image_size( s ):
    num = list( map( int , s.split(',')))
    return num[0] , num[1]

def get_spacing( s ):
    num = list( map( float , s.split(',')))
    return num[0] , num[1] , num[2]


def get_z_position( df ):
    s = df.loc['Normalized_lesion_location']
    num = list( map( float , s.split(',')))
    return num[2]
    
def get_slice_no( df ):
    s = df.loc['Key_slice_index']
    return int(s)

def get_windows( df ):
    s = df.loc[ 'DICOM_windows']
    num = list( map( float , s.split(',')))
    return num


def get_segmentation():
    return []

def get_bbox( df ):
    s = df.loc['Bounding_boxes']
    num = list( map( float , s.split(',')))
    num[2] = num[2]-num[0]
    num[3] = num[3]-num[1]
    return num 

def get_noise( df ):
    s = df.loc['Possibly_noisy']
    num = int(s)
    return num

def get_area( df ):
    s = df.loc['Lesion_diameters_Pixel_']
    num = list( map( float , s.split(',')))
    return num[0]*num[1]
    



newcats = [{'supercategory': 'DeepLesion', 'id': 1, 'name': 'abdomen'},
           {'supercategory': 'DeepLN', 'id': 2, 'name': 'abdomen LN'},
           {'supercategory': 'DeepLesion', 'id': 3, 'name': 'adrenal'},
           {'supercategory': 'DeepLN', 'id': 4, 'name': 'axillary LN'},
           {'supercategory': 'DeepLesion', 'id': 5, 'name': 'bone'},
           {'supercategory': 'DeepLN', 'id': 6, 'name': 'inguinal LN'},
           {'supercategory': 'DeepLesion', 'id': 7, 'name': 'kidney'},
           {'supercategory': 'DeepLesion', 'id': 8, 'name': 'liver'},
           {'supercategory': 'DeepLesion', 'id': 9, 'name': 'lung'},
           {'supercategory': 'DeepLN', 'id': 10, 'name': 'mediastinum LN'},
           {'supercategory': 'DeepLN', 'id': 11, 'name': 'neck LN'},
           {'supercategory': 'DeepLesion', 'id': 12, 'name': 'ovary'},
           {'supercategory': 'DeepLesion', 'id': 13, 'name': 'pancreas'},
           {'supercategory': 'DeepLN', 'id': 14, 'name': 'pelvic LN'},
           {'supercategory': 'DeepLesion', 'id': 15, 'name': 'pelvis'},
           {'supercategory': 'DeepLesion', 'id': 16, 'name': 'pleural'},
           {'supercategory': 'DeepLN', 'id': 17, 'name': 'retroperitoneal LN'},
           {'supercategory': 'DeepLesion', 'id': 18, 'name': 'soft tissue'},
           {'supercategory': 'DeepLesion', 'id': 19, 'name': 'spleen'},
           {'supercategory': 'DeepLesion', 'id': 20, 'name': 'stomach'},
           {'supercategory': 'DeepLesion', 'id': 21, 'name': 'thyroid'} ]

def get_21_lesion_location_cls():
    D_cls = {}
    for d in newcats:
        id_ = d['id']
        name = d['name']
        D_cls[name] = id_
    return D_cls

D_cls = get_21_lesion_location_cls()

def get_category_id( location , Dict ):
    return Dict[location]


# In[4]:

def replace_png_path(s):
    cs = s.replace('AutoRecist/Inputs' , 'AutoRecist/Pngs')
    return cs


# In[5]:

import os
import cv2
import json, yaml
import numpy as np
from PIL import Image
from collections import OrderedDict
from pycocotools import mask as cocomask
from pycocotools import coco as cocoapi


class DeepLesion():
    """
        DL class to convert annotations to COCO Json format
    """
    def __init__(self, df,image_id_start=0,annotation_id_start=0, savename='a.json'):
        self.image_id_start = image_id_start
        self.annotation_id_start = annotation_id_start
        self.df = df 
        self.info = {"year" : 2021,
                     "version" : "2.0",
                     "description" : "Covert Weasis to Json format",
                     "contributor" : "HY,JM,BZ,LS,FSA",
                     "url" : "http:// /",
                     "date_created" : "20211129"
                    }
        self.licenses = [{"id": 1,
                          "name": "Attribution-NonCommercial",
                          "url": "http:// /"
                         }]

        self.categories = newcats
        
        self.images, self.annotations = self.__get_image_annotation_pairs__(self.df)
        json_data = {"info" : self.info,
                     "images" : self.images,
                     "licenses" : self.licenses,
                     "annotations" : self.annotations,
                     "categories" : self.categories}

        with open(savename, "w") as jsonfile:
            json.dump(json_data, jsonfile, sort_keys=True, indent=4)
            
    def change_df(self , df , savename = 'temp.json'):
        self.df = df 

        self.images, self.annotations = self.__get_image_annotation_pairs__(self.df)
        json_data = {"info" : self.info,
                     "images" : self.images,
                     "licenses" : self.licenses,
                     "annotations" : self.annotations,
                     "categories" : self.categories}

        with open(savename, "w") as jsonfile:
            json.dump(json_data, jsonfile, sort_keys=True, indent=4)
            print( 'Saved %s'%savename )
        
            
    def __get_image_annotation_pairs__(self,df):
        images = []
        annotations = []
        self.file_name_dict = {}
        for i , row in df.iterrows():
            try:
                print(i)
                df_image = get_dicom_header_df( row['Image File Path'] )
                png_folder = replace_png_path(row['Image File Path'] )
                
                for one in df_image.index.values.tolist():
#                     file_name = InstanceNumber2file_name(df_image, one)
#                     file_name = os.path.join( row['Image File Path'] , file_name)
                    file_name = os.path.join(png_folder, '%03d.png'%one)
                    file_name = file_name.replace('/mnt/fast-disk1/mjc/AutoRecist/','')

                    if file_name in self.file_name_dict:
                        oneimageid = self.file_name_dict[file_name]
                    else:
                        oneimage = {}
                        oneimage['file_name'] = file_name
                        self.image_id_start += 1
                        oneimageid = self.image_id_start
                        oneimage['id'] = oneimageid

                        oneimage['height'] , oneimage['width'] = int(InstanceNumber2data_element(df_image,one,'Rows')), int( InstanceNumber2data_element(df_image,one,'Columns') )

                        oneimage['slice_no'] = int(one)
                        oneimage['spacing'] = float( InstanceNumber2data_element(df_image,one,'PixelSpacing_0') )
                        oneimage['slice_intv'] = float( get_SliceThickness(df_image) )
                        oneimage['z_position'] = 0.5
                        oneimage['windows'] = InstanceNumber2windows_min_max(df_image,one)

                        images.append(oneimage)
                        self.file_name_dict[file_name] = oneimageid


            except Exception as e: print(e)
        
        return images, annotations
            
    



# def pd_str_replace(df , col, ori, new):
#     if isinstance(col , str):
#         df[col] = df[col].str.replace(ori,new, case = False) 
#     elif isinstance(col, list):
#         for one in col:
#             pd_str_replace(df , one, ori, new)
#     else:
#         raise('col instance should be str or list')

# pd_str_replace(df, ['Image File Path' ], "X:" , "/mnt/X-drive")
# pd_str_replace(df, ['Image File Path' ], r"\\" , "/")
# pd_str_replace(df, ['Image File Path'], "/mnt/X-drive/ClinicalTrialDone/FNIH_VOLPACK", "/mnt/fast-disk1/mjc/AutoRecist/Inputs")
# pd_str_replace(df, ['Image File Path'], "/mnt/X-drive/ClinicalTrials", "/mnt/fast-disk1/mjc/AutoRecist/Inputs")


print('Initial Image Process')
dataset = DeepLesion(df,savename='/mnt/fast-data/mjc/AutoRECIST/Annotations/inference.json')
print('Image Process is Done')
print('Total of {} slice images was Processed.'.format(len(dataset.images)))




