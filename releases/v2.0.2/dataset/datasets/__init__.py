from .brightfiled import Brightfield_Dataset
from .brightfiled_coco import BaseCOCODataset
from .original_plus_synthetic_brightfield import OriginalPlusSyntheticBrightfield
from .rectangle import Rectangle_Dataset

from .evican2 import EVICAN2
from .livecell import LiveCell, LiveCell2Percent, LiveCell30Images
from .yeastnet import YeastNet
from .hubmap import HuBMAP


__all__ = ['Brightfield_Dataset', 'OriginalPlusSyntheticBrightfield', 'Rectangle_Dataset',
           'EVICAN2', 'LiveCell', 'LiveCell2Percent', 'LiveCell30Images', 
           'YeastNet', 'HuBMAP', 'BaseCOCODataset']

# from pycocotools.coco import COCO
# import json
# import re
# import pandas as pd
# import numpy as np
# from os.path import join

# from configs import cfg


# coco = COCO(join(cfg.coco_dataset))
# img_dir = join(cfg.coco_dataset, 'images')

# _json = open(join(cfg.coco_dataset))
# json_data = json.load(_json)


# df = pd.read_csv(cfg.csv_dataset_dir)[:100]  # dummy beffer crop - some images were skipped when labeling 
# df.index = np.arange(0, len(df))
# df['mask_id'] = 0
# df['fl_name'] = 0

# # --------------------
# # Preprocess dataframe
# for i in range(28):
#     image_name = json_data['images'][i]['file_name'].split('-')[-1]
#     image_name = image_name.split('_')[0]

#     encoded_image_name = re.split('(\d+)', image_name.split('_')[0])
#     df.loc[
#         (df['Row'] == int(encoded_image_name[1])) & 
#         (df['Col'] == int(encoded_image_name[3])) & 
#         (df['FieldID'] == int(encoded_image_name[5])),
#         'mask_id'
#     ] = i + 1


#     df.loc[
#         (df['Row'] == int(encoded_image_name[1])) & 
#         (df['Col'] == int(encoded_image_name[3])) & 
#         (df['FieldID'] == int(encoded_image_name[5])),
#         'fl_name'
#     ] = image_name

# df = df.drop(df[df['mask_id'] == 0].index)
# df['mask_id'] -= 1

# df.index = np.arange(0, len(df))
# df['id'] = df.index

