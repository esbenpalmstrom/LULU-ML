#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 20:17:46 2019

@author: esben

Parts of this program is forked from the eo-learn examples found at: https://github.com/sentinel-hub/eo-learn
"""

#%% Imports

# Basics of Python data handling and visualization
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from shapely.geometry import Polygon
from matplotlib import pylab as pl
from matplotlib.patches import Patch

# Basics of GIS
import geopandas as gpd

# The core of this example
from eolearn.core import EOTask, EOPatch, LinearWorkflow, FeatureType, OverwritePermission, LoadFromDisk, SaveToDisk, EOExecutor
from eolearn.io import S2L1CWCSInput, ExportToTiff
from eolearn.mask import AddCloudMaskTask, get_s2_pixel_cloud_detector, AddValidDataMaskTask
from eolearn.geometry import VectorToRaster, PointSamplingTask, ErosionTask
from eolearn.features import LinearInterpolation, SimpleFilterTask
from sentinelhub import BBoxSplitter, BBox, CRS, CustomUrlParam

# Machine learning 
import lightgbm as lgb
from sklearn.externals import joblib
from sklearn import metrics
from sklearn import preprocessing

# Misc
import pickle
import sys
import os
import datetime
import itertools
from tqdm import tqdm_notebook as tqdm
import enum

import rasterio as rio
from rasterio.plot import show
from rasterio.plot import show_hist
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from fiona.crs import from_epsg
import pycrs

import seaborn as sns;
import random
import math

#%% Section 1
# Selection of patch IDs

#Short script to select a random set of patches for training and testing
#Select interval of tiles.
#All tiles will be: patchIDs = np.arange(1,618)
full_patchIDs = np.arange(1,618)
patch_subset_amount = 20
patchIDs = np.random.choice(full_patchIDs,patch_subset_amount,False)

patchIDs = [39,45,90,104,144,151,185,220,260,299,335,343,348,375,391,420,475,507,531,593] #Uncomment this line to select patches manually

#patchIDs = np.linspace(0,618,num=20,endpoint=False,dtype=int)#Uncomment this line for a list of evenly spaced numbers. Use np.linspace


#For a random set of Patches for training.
train_len = math.floor(len(patchIDs)*0.8)
test_len = len(patchIDs)-train_len
np.random.seed(0)
train_patchIDs = np.random.choice(patchIDs,train_len,False)

#Adding patches not in train to test
test_patchIDs =[]
for i in range(len(patchIDs)):
    if (patchIDs[i] not in train_patchIDs):
        test_patchIDs.append(patchIDs[i])

test_patchIDs = np.int64(test_patchIDs) # Is this even necessary??

#TIME INTERVAL
time_interval = ['2017-01-01', '2017-12-31'] # time interval for the sentinelhub request



#%% Section 2
#Defining AOI and making Bboxes

country = gpd.read_file('./DK.geojson')

# Convert CRS to UTM_32N
country_crs = CRS.UTM_32N
country = country.to_crs(crs={'init': CRS.ogc_string(country_crs)})

# Get the country's shape in polygon format
country_shape = country.geometry.values.tolist()[-1]
#country.plot()
#plt.axis('off');


# Create the splitter to obtain a list of bboxes
bbox_splitter_large = BBoxSplitter([country_shape], country_crs, (45, 35))
bbox_splitter_small = BBoxSplitter([country_shape], country_crs, (45 * 3, 35 * 3))

bbox_splitter = bbox_splitter_large

bbox_list = np.array(bbox_splitter.get_bbox_list())
info_list = np.array(bbox_splitter.get_info_list())


#Prepare info of selected patch IDs
geometry = [Polygon(bbox.get_polygon()) for bbox in bbox_list[patchIDs]]
idxs_x = [info['index_x'] for info in info_list[patchIDs]]
idxs_y = [info['index_y'] for info in info_list[patchIDs]]
df = pd.DataFrame({'index_x': idxs_x, 'index_y': idxs_y})
gdf = gpd.GeoDataFrame(df, 
                       crs={'init': CRS.ogc_string(country_crs)}, 
                       geometry=geometry)

#Figure of selected patches
fig, ax = plt.subplots(figsize=(10, 10))
gdf.plot(ax=ax,facecolor='w',edgecolor='r',alpha=0.5)
country.plot(ax=ax, facecolor='w',edgecolor='b',alpha=0.5)
ax.set_title('Selected patches from Denmark (45 x 35 grid)');
plt.axis('off')
plt.savefig('Selected_patches.png')

shapefile_name =  './selected_bboxes_Denmark_large.shp'
gdf.to_file(shapefile_name)

#%% 2.1

poly = gdf['geometry'][0]
x1, y1, x2, y2 = poly.bounds
aspect_ratio = (y1 - y2) / (x1 - x2)

# content of the geopandas dataframe
gdf.head()

#%% Section 3
#Defining classes and tasks needed for filling EOPatches with data.

#Define some needed custom EOTasks
class SentinelHubValidData:
    """
    Combine Sen2Cor's classification map with `IS_DATA` to define a `VALID_DATA_SH` mask
    The SentinelHub's cloud mask is asumed to be found in eopatch.mask['CLM']
    """
    def __call__(self, eopatch):
        return np.logical_and(eopatch.mask['IS_DATA'].astype(np.bool),
                              np.logical_not(eopatch.mask['CLM'].astype(np.bool)))

class CountValid(EOTask):
    """
    The task counts number of valid observations in time-series and stores the results in the timeless mask.
    """
    def __init__(self, count_what, feature_name):
        self.what = count_what
        self.name = feature_name

    def execute(self, eopatch):
        eopatch.add_feature(FeatureType.MASK_TIMELESS, self.name, np.count_nonzero(eopatch.mask[self.what],axis=0))

        return eopatch


class NormalizedDifferenceIndex(EOTask):
    """
    The tasks calculates user defined Normalised Difference Index (NDI) between two bands A and B as:
    NDI = (A-B)/(A+B).
    """
    def __init__(self, feature_name, band_a, band_b):
        self.feature_name = feature_name
        self.band_a_fetaure_name = band_a.split('/')[0]
        self.band_b_fetaure_name = band_b.split('/')[0]
        self.band_a_fetaure_idx = int(band_a.split('/')[-1])
        self.band_b_fetaure_idx = int(band_b.split('/')[-1])

    def execute(self, eopatch):
        band_a = eopatch.data[self.band_a_fetaure_name][..., self.band_a_fetaure_idx]
        band_b = eopatch.data[self.band_b_fetaure_name][..., self.band_b_fetaure_idx]

        ndi = (band_a - band_b) / (band_a  + band_b)

        eopatch.add_feature(FeatureType.DATA, self.feature_name, ndi[..., np.newaxis])

        return eopatch


class EuclideanNorm(EOTask):
    """
    The tasks calculates Euclidian Norm of all bands within an array:
    norm = sqrt(sum_i Bi**2),
    where Bi are the individual bands within user-specified feature array.
    """
    def __init__(self, feature_name, in_feature_name):
        self.feature_name = feature_name
        self.in_feature_name = in_feature_name

    def execute(self, eopatch):
        arr = eopatch.data[self.in_feature_name]
        norm = np.sqrt(np.sum(arr**2, axis=-1))

        eopatch.add_feature(FeatureType.DATA, self.feature_name, norm[..., np.newaxis])
        return eopatch


#%% For RefMap2

class LULC(enum.Enum):
    IKKE_KORTLAGT       = (0,'No data','black')
    BEBYGGELSE          = (1,'Artificial surface','crimson')
    SKOV                = (2,'Forest','xkcd:darkgreen')
    LANDBRUG            = (3,'Agriculture','xkcd:lime')
    AABEN_NATUR         = (4,'Open nature','xkcd:tan')
    VAND                = (5,'Water','xkcd:azure')


    def __init__(self, val1, val2, val3):
        self.id = val1
        self.class_name = val2
        self.color = val3

feature=(FeatureType.MASK_TIMELESS, 'LULC')

lulc_cmap = mpl.colors.ListedColormap([entry.color for entry in LULC])
lulc_norm = mpl.colors.BoundaryNorm(np.arange(-0.5, 6, 1), lulc_cmap.N)

#%%


# TASK FOR BAND DATA
# add a request for B(B02), G(B03), R(B04), NIR (B08), SWIR1(B11), SWIR2(B12)
# from default layer 'ALL_BANDS' at 10m resolution
# Here we also do a simple filter of cloudy scenes. A detailed cloud cover
# detection is performed in the next step
custom_script = 'return [B02, B03, B04, B08, B11, B12];'
add_data = S2L1CWCSInput(
    layer='BANDS-S2-L1C',
    feature=(FeatureType.DATA, 'BANDS'), # save under name 'BANDS'
    custom_url_params={CustomUrlParam.EVALSCRIPT: custom_script}, # custom url for 6 specific bands
    resx='10m', # resolution x
    resy='10m', # resolution y
    maxcc=0.8, # maximum allowed cloud cover of original ESA tiles
)

# TASK FOR CLOUD INFO
# cloud detection is performed at 80m resolution
# and the resulting cloud probability map and mask
# are scaled to EOPatch's resolution
cloud_classifier = get_s2_pixel_cloud_detector(average_over=2, dilation_size=1, all_bands=False)
add_clm = AddCloudMaskTask(cloud_classifier, 'BANDS-S2CLOUDLESS', cm_size_y='80m', cm_size_x='80m',
                           cmask_feature='CLM', # cloud mask name
                           cprobs_feature='CLP' # cloud prob. map name
                          )

# TASKS FOR CALCULATING NEW FEATURES
# NDVI: (B08 - B04)/(B08 + B04)
# NDWI: (B03 - B08)/(B03 + B08)
# NORM: sqrt(B02^2 + B03^2 + B04^2 + B08^2 + B11^2 + B12^2)
ndvi = NormalizedDifferenceIndex('NDVI', 'BANDS/3', 'BANDS/2')
ndwi = NormalizedDifferenceIndex('NDWI', 'BANDS/1', 'BANDS/3')
norm = EuclideanNorm('NORM','BANDS')

# TASK FOR VALID MASK
# validate pixels using SentinelHub's cloud detection mask and region of acquisition
add_sh_valmask = AddValidDataMaskTask(SentinelHubValidData(),
                                      'IS_VALID' # name of output mask
                                     )

# TASK FOR COUNTING VALID PIXELS
# count number of valid observations per pixel using valid data mask
count_val_sh = CountValid('IS_VALID', # name of existing mask
                          'VALID_COUNT' # name of output scalar
                         )

# TASK FOR SAVING TO OUTPUT (if needed)
path_out = './eopatches_large/'
if not os.path.isdir(path_out):
    os.makedirs(path_out)
save = SaveToDisk(path_out, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)


#%% Section 4
#Execution of workflow: Filling patches with data
print('Starting download')

workflow = LinearWorkflow(
    add_data,
    add_clm,
    ndvi,
    ndwi,
    norm,
    add_sh_valmask,
    count_val_sh,
    save
)

for idx, bbox in enumerate(bbox_list[patchIDs]):

    # define additional parameters of the workflow
    extra_param = {
        add_data:{'bbox': bbox, 'time_interval': time_interval},
        save: {'eopatch_folder': 'eopatch_{}'.format(idx)}
    }

    workflow.execute(extra_param)

print('Download finished')

#%% Check the IS_VALID npy array

#isvalid = np.load('./eopatches_large/eopatch_0/mask/IS_VALID.npy')

#print(np.sum(isvalid))
#print(np.size(isvalid) - np.count_nonzero(isvalid))

#%% See the structure of a selected EOPatch


EOPatch.load('./eopatches_large/eopatch_3/')


#%% Section 5
#Clip reference map with selected EOPach polygons and add the clip to each EOPatch

# Ref as in https://www.earthdatascience.org/courses/earth-analytics-python/lidar-raster-data/classify-plot-raster-data-in-python/
#and in : https://www.earthdatascience.org/courses/earth-analytics-python/lidar-raster-data/customize-matplotlib-raster-maps/


#Loading raster and getting coordinates and crs data.
land_cover_path = './reference_map/RefMap2.tif'
raster = rio.open(land_cover_path)



gdf = gdf.to_crs(crs=raster.crs.data)
def getFeatures(gdf,n):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    return [json.loads(gdf.to_json())['features'][n]['geometry']]
epsg_code = int(raster.crs.data['init'][5:])


#Loop to fill all EOPatches
for i in range(len(patchIDs)):
    coords = getFeatures(gdf,i)
    print(coords)
    out_img, out_transform = mask(dataset=raster, shapes=coords, crop=True) #Fitting raster into shape
    out_meta = raster.meta.copy()
    print(out_meta)
    out_meta.update({"driver": "GTiff",
                     "height": out_img.shape[1],
                     "width": out_img.shape[2],
                     "transform": out_transform,
                     "crs": pycrs.parse.from_epsg_code(epsg_code).to_proj4()}
                         )
    out_tif = os.path.join('./masked_ref/masked_{}.tif'.format(i))
    with rio.open(out_tif, "w", **out_meta) as dest:
            dest.write(out_img)
    
    arr2 = []
    
    arr2 = np.swapaxes(out_img,0,1)
    arr2 = np.swapaxes(arr2,1,2)
    
    if arr2.shape == (1005,999,1):
        arr2 = np.delete(arr2,0,axis=1)
        arr2 = np.delete(arr2,0,axis=0)        
    elif arr2.shape == (1005,1000,1):
        arr2 = np.delete(arr2,0,axis=1)
        arr2 = np.delete(arr2,0,axis=1)
        arr2 = np.delete(arr2,0,axis=0)
    elif arr2.shape == (1004,999,1):
        arr2 = np.delete(arr2,0,axis=1)
    elif arr2.shape == (1004,1000,1):
        arr2 = np.delete(arr2,0,axis=1)
        arr2 = np.delete(arr2,0,axis=1)
    
    if arr2.shape ==(1004,998,1):
        print('Correct size')
    else:
        print('WRONG SIZE!')
    
    np.save('./eopatches_large/eopatch_{}/mask_timeless/LULC.npy'.format(i),arr2)
    
    
    

#clipped = rio.open(out_tif) # Open the last clipped raster file
# Open the clipped raster file
clipped = rio.open(out_tif)



#%%Plotting reference from np array

with rio.open('./masked_ref/masked_14.tif', 'r') as ds:
    arr = ds.read(1)  # read all raster values

fig, ax = plt.subplots(figsize=(13,9))
im = ax.imshow(arr,
               cmap=lulc_cmap,
               norm=lulc_norm)
ax.set_title('Reference map')

legend_labels = {'black':'No data',
                 'crimson':'Artificial surface',
                 'xkcd:darkgreen':'Forest',
                 'xkcd:lime':'Agriculture',
                 'xkcd:tan':'Open nature',
                 'xkcd:azure':'Water',
                 }

patches = [Patch(color=color, label=label) for color, label in legend_labels.items()]
ax.legend(handles=patches, 
          bbox_to_anchor=(1.25,1), 
          facecolor="white")
ax.set_axis_off()
plt.show()


#%% Visualize reference map from LULC.npy to check if it works..

path_out = './eopatches_large/'

fig = plt.figure(figsize=(10, 10))
eopatch = EOPatch.load('./eopatches_large/eopatch_0')
plt.imshow(eopatch.mask_timeless['LULC'].squeeze(), cmap=lulc_cmap, norm=lulc_norm)

#%% Section 6
# Filtering and stuff.

class ConcatenateData(EOTask):
    """
        Task to concatenate data arrays along the last dimension
    """
    def __init__(self, feature_name, feature_names_to_concatenate):
        self.feature_name = feature_name
        self.feature_names_to_concatenate = feature_names_to_concatenate

    def execute(self, eopatch):
        arrays = [eopatch.data[name] for name in self.feature_names_to_concatenate]

        eopatch.add_feature(FeatureType.DATA, self.feature_name, np.concatenate(arrays, axis=-1))

        return eopatch


class ValidDataFractionPredicate:
    """
    Predicate that defines if a frame from EOPatch's time-series is valid or not. Frame is valid, if the
    valid data fraction is above the specified threshold.
    """
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, array):
        coverage = np.sum(array.astype(np.uint8)) / np.prod(array.shape)
        return coverage > self.threshold

#%%

# TASK TO LOAD EXISTING EOPATCHES
load = LoadFromDisk(path_out)

# TASK FOR CONCATENATION
concatenate = ConcatenateData('FEATURES', ['BANDS', 'NDVI', 'NDWI', 'NORM'])

# TASK FOR FILTERING OUT TOO CLOUDY SCENES
# keep frames with > 80 % valid coverage
valid_data_predicate = ValidDataFractionPredicate(0.8)
filter_task = SimpleFilterTask((FeatureType.MASK, 'IS_VALID'), valid_data_predicate)

# TASK FOR LINEAR INTERPOLATION
# linear interpolation of full time-series and date resampling
resampled_range = ('2017-01-01', '2017-12-31', 16)
linear_interp = LinearInterpolation(
    'FEATURES', # name of field to interpolate
    mask_feature=(FeatureType.MASK, 'IS_VALID'), # mask to be used in interpolation
    copy_features=[(FeatureType.MASK_TIMELESS, 'LULC')], # features to keep
    resample_range=resampled_range, # set the resampling range
    bounds_error=False # extrapolate with NaN's
)

# TASK FOR EROSION
# erode each class of the reference map
erosion = ErosionTask(mask_feature=(FeatureType.MASK_TIMELESS,'LULC','LULC_ERODED'), disk_radius=1)

# TASK FOR SPATIAL SAMPLING
# Uniformly sample about pixels from patches
n_samples = int(1e5) # no. of pixels to sample
ref_labels = [0,1,2,3,4,5] # reference labels to take into account when sampling
spatial_sampling = PointSamplingTask(
    n_samples=n_samples,
    ref_mask_feature='LULC_ERODED',
    ref_labels=ref_labels,
    sample_features=[  # tag fields to sample
        (FeatureType.DATA, 'FEATURES'),
        (FeatureType.MASK_TIMELESS, 'LULC_ERODED')
    ])

path_out_sampled = './eopatches_sampled_large/'
if not os.path.isdir(path_out_sampled):
    os.makedirs(path_out_sampled)
save = SaveToDisk(path_out_sampled, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)


#%%


# Define the workflow
workflow = LinearWorkflow(
    load,
    concatenate,
    filter_task,
    linear_interp,
    erosion,
    spatial_sampling,
    save
)

#%% Section 6.1
#Preparation for ML: Running workflow

print('Starting data processing')
# Execute the workflow
for i in range(len(patchIDs)):
    # define additional parameters of the workflow
    extra_param = {
        load: {'eopatch_folder': 'eopatch_{}'.format(i)},
        save: {'eopatch_folder': 'eopatch_{}'.format(i)}
    }
    
    workflow.execute(extra_param)

print('Data processing finished')
#%% True-color plot (Be wary of clouds)
fig = plt.figure(figsize=(10, 10))
eopatch = EOPatch.load('./eopatches_large/eopatch_14')
plt.imshow(np.clip(eopatch.data['BANDS'].squeeze()[57][..., [2, 1, 0]] * 3.5, 0, 1)) 
"""
For above plotting function:
First square bracket: Data source
Second square bracket: change picture instance/time frame
Third square bracket: Band combination
"""
plt.xticks([])
plt.yticks([])
fig.subplots_adjust(wspace=0, hspace=0)

#%% True color plot loop - Uncomment to display all the images in a certain patch.
#for i in range(61):
#    fig = plt.figure(figsize=(10, 10))
#    eopatch = EOPatch.load('./eopatches_large/eopatch_14')
#    plt.imshow(np.clip(eopatch.data['BANDS'].squeeze()[i][..., [2, 1, 0]] * 3.5, 0, 1)) 
#    plt.xticks([])
#    plt.yticks([])
#    fig.subplots_adjust(wspace=0, hspace=0)

#%% True-color plot (Be wary of clouds)
fig = plt.figure(figsize=(10, 10))
eopatch = EOPatch.load('./eopatches_large/eopatch_5')
plt.imshow(np.clip(eopatch.data['BANDS'].squeeze()[0][..., [5, 1, 0]] * 3.5, 0, 1)) 
"""
For above plotting function:
First square bracket: Data source
Second square bracket: change picture instance/time frame
Third square bracket: Band combination
"""
plt.xticks([])
plt.yticks([])
fig.subplots_adjust(wspace=0, hspace=0)
#%% Section 7: Training model

eopatches = []
path_out_sampled = './eopatches_sampled_large/'

for i in range(len(patchIDs)):
    eopatches.append(EOPatch.load('{}/eopatch_{}'.format(path_out_sampled, i), lazy_loading=True))

eopatches = np.array(eopatches)

#%%

# Definition of the train and test patch IDs
length = np.arange(len(patchIDs))
train_ID = np.random.choice(length,size=len(train_patchIDs),replace=False)
#test_ID = np.arange(train_ID[len(train_ID-1)],)

test_ID =[]
for i in range(len(length)):
    if (length[i] not in train_ID):
        test_ID.append(length[i])

#%%
# Set the features and the labels for train and test sets
features_train = np.array([eopatch.data['FEATURES_SAMPLED'] for eopatch in eopatches[train_ID]])
labels_train = np.array([eopatch.mask_timeless['LULC_ERODED_SAMPLED'] for eopatch in eopatches[train_ID]])
features_test = np.array([eopatch.data['FEATURES_SAMPLED'] for eopatch in eopatches[test_ID]])
labels_test = np.array([eopatch.mask_timeless['LULC_ERODED_SAMPLED'] for eopatch in eopatches[test_ID]])

# get shape
p1, t, w, h, f = features_train.shape
p2, t, w, h, f = features_test.shape
p = p1 + p2

# reshape to n x m
features_train = np.moveaxis(features_train, 1, 3).reshape(p1 * w * h, t * f)
labels_train = np.moveaxis(labels_train, 1, 2).reshape(p1 * w * h, 1).squeeze()
features_test = np.moveaxis(features_test, 1, 3).reshape(p2 * w * h, t * f)
labels_test = np.moveaxis(labels_test, 1, 2).reshape(p2 * w * h, 1).squeeze()

# remove points with no reference from training (so we dont train to recognize "no data")
mask_train = labels_train == 0
features_train = features_train[~mask_train]
labels_train = labels_train[~mask_train]

# remove points with no reference from test (so we dont validate on "no data", which doesn't make sense)
mask_test = labels_test == 0
features_test = features_test[~mask_test]
labels_test = labels_test[~mask_test]

#%% Set up and train model (Skip this step if model has already been made)

# Set up training classes
labels_unique = np.unique(labels_train)

# Set up the model
model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=len(labels_unique),
    metric='multi_logloss'
)

# train the model
model.fit(features_train, labels_train)

# uncomment to save the model
model_base_name = 'model_SI_LULC_larger'
joblib.dump(model, './{}.pkl'.format(model_base_name))

#%% Section 8: Model validation and evaluation

# uncomment to load the model and replace with your file, usually just correct the date
model_path = './model_SI_LULC_larger.pkl'
model = joblib.load(model_path)

# predict the test labels
plabels_test = model.predict(features_test)

#%% OA and weighted F1 score
#https://medium.com/thalus-ai/performance-metrics-for-classification-problems-in-machine-learning-part-i-b085d432082b
#See above for F1 explanation


print('Classification accuracy {:.1f}%'.format(100 * metrics.accuracy_score(labels_test, plabels_test)),file=open("accuracy.txt","w"))
print('Classification F1-score {:.1f}%'.format(100 * metrics.f1_score(labels_test, plabels_test, average='weighted')),file=open("accuracy.txt","a"))



#%% For each class seperately

class_labels = np.unique(labels_test)
class_names = [entry.class_name for entry in LULC]

f1_scores = metrics.f1_score(labels_test, plabels_test, labels=class_labels, average=None)
recall = metrics.recall_score(labels_test, plabels_test, labels=class_labels, average=None)
precision = metrics.precision_score(labels_test, plabels_test, labels=class_labels, average=None)

print('             Class              =  F1  | Recall | Precision',file=open("accuracy.txt","a"))
print('         --------------------------------------------------',file=open("accuracy.txt","a"))
for idx, lulctype in enumerate([class_names[idx] for idx in class_labels]):
    print('         * {0:20s} = {1:2.1f} |  {2:2.1f}  | {3:2.1f}'.format(lulctype,
                                                                         f1_scores[idx] * 100,
                                                                         recall[idx] * 100,
                                                                         precision[idx] * 100),file=open("accuracy.txt","a"))

#%% Standard and transposed confusion matrices

# Define the plotting function
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues, ylabel='True label', xlabel='Predicted label', filename=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    np.set_printoptions(precision=2, suppress=True)

    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + np.finfo(np.float).eps)

    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    plt.title(title, fontsize=10)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=12)

    plt.tight_layout()
    plt.ylabel(ylabel, fontsize=10)
    plt.xlabel(xlabel, fontsize=10)

#%% Standard and transposed confusion matrices
fig = plt.figure(figsize=(10, 9))

plt.subplot(1, 2, 1)
conf_matrix_gbm = metrics.confusion_matrix(labels_test, plabels_test)
plot_confusion_matrix(conf_matrix_gbm,
                      classes=[name for idx, name in enumerate(class_names) if idx in class_labels],
                      normalize=True,
                      ylabel='Truth (LAND COVER)',
                      xlabel='Predicted (GBM)',
                      title='Confusion matrix');

plt.subplot(1, 2, 2)
conf_matrix_gbm = metrics.confusion_matrix(plabels_test, labels_test)
plot_confusion_matrix(conf_matrix_gbm,
                      classes=[name for idx, name in enumerate(class_names) if idx in class_labels],
                      normalize=True,
                      xlabel='Truth (LAND COVER)',
                      ylabel='Predicted (GBM)',
                      title='Transposed Confusion matrix');

plt.tight_layout()

#%% Just the transposed confusion matrix.
fig = plt.figure(figsize=(10, 9))

plt.subplot(1, 1, 1)
conf_matrix_gbm = metrics.confusion_matrix(plabels_test, labels_test)
plot_confusion_matrix(conf_matrix_gbm,
                      classes=[name for idx, name in enumerate(class_names) if idx in class_labels],
                      normalize=True,
                      xlabel='Truth (LAND COVER)',
                      ylabel='Predicted (GBM)',
                      title='Transposed Confusion matrix');

#%% Visualization of results
class PredictPatch(EOTask):
    """
    Task to make model predictions on a patch. Provide the model and the feature,
    and the output names of labels and scores (optional)
    """
    def __init__(self, model, features_feature, predicted_labels_name, predicted_scores_name=None):
        self.model = model
        self.features_feature = features_feature
        self.predicted_labels_name = predicted_labels_name
        self.predicted_scores_name = predicted_scores_name

    def execute(self, eopatch):
        ftrs = eopatch[self.features_feature[0]][self.features_feature[1]]

        t, w, h, f = ftrs.shape
        ftrs = np.moveaxis(ftrs, 0, 2).reshape(w * h, t * f)

        plabels = self.model.predict(ftrs)
        plabels = plabels.reshape(w, h)
        plabels = plabels[..., np.newaxis]
        eopatch.add_feature(FeatureType.MASK_TIMELESS, self.predicted_labels_name, plabels)

        if self.predicted_scores_name:
            pscores = self.model.predict_proba(ftrs)
            _, d = pscores.shape
            pscores = pscores.reshape(w, h, d)
            eopatch.add_feature(FeatureType.DATA_TIMELESS, self.predicted_scores_name, pscores)

        return eopatch

#%%
# TASK TO LOAD EXISTING EOPATCHES
load = LoadFromDisk(path_out_sampled)

# TASK FOR PREDICTION
predict = PredictPatch(model, (FeatureType.DATA, 'FEATURES'), 'LBL_GBM', 'SCR_GBM')

# TASK FOR SAVING
save = SaveToDisk(str(path_out_sampled), overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

# TASK TO EXPORT TIFF
export_tiff = ExportToTiff((FeatureType.MASK_TIMELESS, 'LBL_GBM'))
tiff_location = './predicted_tiff'
if not os.path.isdir(tiff_location):
    os.makedirs(tiff_location)

workflow = LinearWorkflow(
    load,
    predict,
    export_tiff,
    save
)

#%%
# create a list of execution arguments for each patch
execution_args = []
for i in range(len(patchIDs)):
    execution_args.append(
        {
            load: {'eopatch_folder': 'eopatch_{}'.format(i)},
            export_tiff: {'filename': '{}/prediction_eopatch_{}.tiff'.format(tiff_location, i)},
            save: {'eopatch_folder': 'eopatch_{}'.format(i)}
        }
    )

# run the executor on 2 cores
executor = EOExecutor(workflow, execution_args)
executor.run(workers = 2)

# uncomment below save the logs in the current directory and produce a report!
#executor = EOExecutor(workflow, execution_args, save_logs=True)
#executor.run(workers = 2)
#executor.make_report()


#%%Plot: Frequency of classes

fig = plt.figure(figsize=(15, 8))

label_ids, label_counts = np.unique(labels_train, return_counts=True)

plt.bar(range(len(label_ids)), label_counts)
plt.xticks(range(len(label_ids)), [class_names[i] for i in label_ids], rotation=45, fontsize=10);
plt.yticks(fontsize=10);

#%% Plot: Band and feature importance
# names of features
fnames = ['B2','B3','B4','B8','B11','B12','NDVI','NDWI','NORM']

# get feature importances and reshape them to dates and features
z = np.zeros(t * f)
z = model.feature_importances_
z = z.reshape((t, f))

fig = plt.figure(figsize=(13, 9))
ax = plt.gca()

# plot the importances
im = ax.imshow(z, aspect=0.25)
plt.xticks(range(len(fnames)), fnames, rotation=45, fontsize=10)
plt.yticks(range(t), ['T{}'.format(i) for i in range(t)], fontsize=10)
plt.xlabel('Bands and band related features', fontsize=10)
plt.ylabel('Time frames', fontsize=10)
ax.xaxis.tick_top()
ax.xaxis.set_label_position('top')

#cax = fig.add_axes([0.82, 0.125, 0.04, 0.755])
#plt.colorbar(im, cax=cax)

fig.subplots_adjust(wspace=0, hspace=0)

cb = fig.colorbar(im, ax=[ax], orientation='horizontal', pad=0.01, aspect=100)
cb.ax.tick_params(labelsize=10)


#%%Plot: Map of valid pixel counts

vmin, vmax = None, None

data = eopatch.mask_timeless['VALID_COUNT'].squeeze()
vmin = np.min(data) if vmin is None else (np.min(data) if np.min(data) < vmin else vmin)
vmax = np.max(data) if vmax is None else (np.max(data) if np.max(data) > vmax else vmax)


fig, axes = plt.subplots(figsize=(10, 10))

im = axes.imshow(eopatch.mask_timeless['VALID_COUNT'].squeeze(), vmin=vmin, vmax=vmax, cmap=plt.cm.inferno)

cb = fig.colorbar(im,orientation='horizontal',aspect=100,pad=0.05)
plt.xticks([])
plt.yticks([])


#%%Plot: Mean of NDVI

fig, axes = plt.subplots(figsize=(10, 10))

ndvi = eopatch.data['NDVI']
mask = eopatch.mask['IS_VALID']
ndvi[~mask] = np.nan
ndvi_mean = np.nanmean(ndvi, axis=0).squeeze()
im = axes.imshow(ndvi_mean, vmin=0, vmax=0.8, cmap=plt.get_cmap('YlGn'))
cb = fig.colorbar(im,orientation='horizontal',aspect=100,pad=0.05)
plt.xticks([])
plt.yticks([])

#%% Plot: Ground truth vs. Prediction

fig = plt.figure(figsize=(16,7))

eopatch = EOPatch.load('./eopatches_sampled_large/eopatch_14',lazy_loading=True)

ax = plt.subplot(1,2,1)
plt.imshow(eopatch.mask_timeless['LULC'].squeeze(), cmap=lulc_cmap, norm=lulc_norm)
plt.xticks([])
plt.yticks([])
ax.set_aspect("auto")
plt.title('Ground Truth', fontsize=10)

ax = plt.subplot(1,2,2)
plt.imshow(eopatch.mask_timeless['LBL_GBM'].squeeze(), cmap=lulc_cmap, norm=lulc_norm)
plt.xticks([])
plt.yticks([])
ax.set_aspect("auto")
plt.title('Prediction', fontsize=10)

legend_labels = {'black':'No data',
                 'crimson':'Artificial surface',
                 'xkcd:darkgreen':'Forest',
                 'xkcd:lime':'Agriculture',
                 'xkcd:tan':'Open nature',
                 'xkcd:azure':'Water',
                 }

patches = [Patch(color=color, label=label) for color, label in legend_labels.items()]
ax.legend(handles=patches, 
          bbox_to_anchor=(1.29,1), 
          facecolor="white")

#%% Plot: Another example of ground truth vs prediction vs true color??
fig = plt.figure(figsize=(4,9))

eopatch = EOPatch.load('./eopatches_sampled_large/eopatch_19',lazy_loading=True)

ax = plt.subplot(3,1,1)
plt.imshow(eopatch.mask_timeless['LULC'].squeeze(), cmap=lulc_cmap, norm=lulc_norm)
plt.xticks([])
plt.yticks([])
ax.set_aspect("auto")
plt.title('Ground Truth', fontsize=10)

ax = plt.subplot(3,1,2)
plt.imshow(eopatch.mask_timeless['LBL_GBM'].squeeze(), cmap=lulc_cmap, norm=lulc_norm)
plt.xticks([])
plt.yticks([])
ax.set_aspect("auto")
plt.title('Prediction', fontsize=10)

legend_labels = {'black':'No data',
                 'crimson':'Artificial surface',
                 'xkcd:darkgreen':'Forest',
                 'xkcd:lime':'Agriculture',
                 'xkcd:tan':'Open nature',
                 'xkcd:azure':'Water',
                 }



ax = plt.subplot(3,1,3)
plt.imshow(np.clip(eopatch.data['FEATURES'][15][..., [2, 1, 0]] * 3.5, 0, 1)) 
plt.xticks([])
plt.yticks([])
ax.set_aspect("auto")
plt.title('True Color', fontsize=10)


fig.subplots_adjust(wspace=0.1, hspace=0.1)

#%% Plot
eopatch = EOPatch.load('./eopatches_sampled_large/eopatch_0',lazy_loading=True)

ax = plt.subplot(2,3,4)
plt.imshow(eopatch.mask_timeless['LULC'].squeeze(), cmap=lulc_cmap, norm=lulc_norm)
plt.xticks([])
plt.yticks([])
ax.set_aspect("auto")
plt.title('Ground Truth', fontsize=10)

ax = plt.subplot(2,3,5)
plt.imshow(eopatch.mask_timeless['LBL_GBM'].squeeze(), cmap=lulc_cmap, norm=lulc_norm)
plt.xticks([])
plt.yticks([])
ax.set_aspect("auto")
plt.title('Prediction', fontsize=10)

ax = plt.subplot(2,3,6)
plt.imshow(np.clip(eopatch.data['FEATURES'][15][..., [2, 1, 0]] * 3.5, 0, 1)) 
plt.xticks([])
plt.yticks([])
ax.set_aspect("auto")
plt.title('True Color', fontsize=10)

#%% Plotting bar diagram of frequency of classes

fig = plt.figure(figsize=(10, 7))

label_ids, label_counts = np.unique(labels_train, return_counts=True)

plt.bar(range(len(label_ids)), label_counts)
plt.xticks(range(len(label_ids)), [class_names[i] for i in label_ids], rotation=45, fontsize=7);
plt.yticks(fontsize=10);