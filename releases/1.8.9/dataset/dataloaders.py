from os.path import join

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from .prepare_dataset import build_dataset
from utils.normalize import normalize
from utils.augmentations import train_transforms, valid_transforms

from configs import cfg


def get_datasets(cfg: cfg, df, fold=0):
    train_df = df.query("fold!=@fold").reset_index(drop=True)
    valid_df = df.query("fold==@fold").reset_index(drop=True)

    dataset = build_dataset(name=cfg.dataset.name)


    train_dataset = dataset(
        df=train_df,
        run_type='train',
        img_size=cfg.train.size,
        normalization=normalize,
        transform=train_transforms(cfg)
    )

    valid_dataset = dataset(
        df=valid_df,
        run_type='valid',
        img_size=cfg.valid.size,
        normalization=normalize,
        transform=valid_transforms(cfg)
    )

    return train_dataset, valid_dataset


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


def get_dataloaders(cfg: cfg, df, fold=0):
    train_dataset, valid_dataset = get_datasets(cfg, df, fold)

    if len(train_dataset) > 0:
        train_loader = DataLoader(train_dataset, batch_size=cfg.train.bs,
                                num_workers=2, collate_fn=trivial_batch_collator, 
                                shuffle=True, pin_memory=True, drop_last=False)
    else:
        train_loader = None
    
    if len(valid_dataset) > 0:
        valid_loader = DataLoader(valid_dataset, batch_size=cfg.valid.bs,
                                num_workers=2, collate_fn=trivial_batch_collator, 
                                shuffle=False, pin_memory=True)
    else:
        valid_loader = None

    return train_loader, valid_loader




if __name__ == '__main__':
    import re
    from .datasets.brightfiled import json_data
    from utils.visualise import visualize
    from utils.utils import flatten_mask

    from .datasets import df

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



    train_dataset = Brightfield_Dataset(
        df=df,
        run_type='train',
        img_size=cfg.train.size,
        normalization=normalize,
        transform=train_transforms(cfg)
    )
    bf, pc, mask = train_dataset[0]
    print(mask.shape)
    print(mask.min(), mask.max())

    # visualize(
    #     [20, 8],
    #     bf_lo=bf[0, ...],
    #     bf_hi=bf[1, ...],
    #     pc=pc[0, ...],
    #     mask=flatten_mask(mask.cpu().detach().numpy(), axis=0)[0, ...],
    #     mask_sample=mask[0, ...]
    # )
    
