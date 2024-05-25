from os.path import join
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset

# from .prepare_dataset import build_dataset
from utils.normalize import normalize
from utils.augmentations import train_transforms, valid_transforms

from configs import cfg


# def get_datasets(cfg: cfg, df, fold=0):
#     train_df = df.query("fold!=@fold").reset_index(drop=True)
#     valid_df = df.query("fold==@fold").reset_index(drop=True)

#     dataset = build_dataset(name=cfg.dataset.name)


#     train_dataset = dataset(
#         df=train_df,
#         run_type='train',
#         img_size=cfg.train.size,
#         normalization=normalize,
#         transform=train_transforms(cfg)
#     )

#     valid_dataset = dataset(
#         df=valid_df,
#         run_type='valid',
#         img_size=cfg.valid.size,
#         normalization=normalize,
#         transform=valid_transforms(cfg)
#     )

#     return train_dataset, valid_dataset


def trivial_batch_collator(batch):
    """
    A batch collator that does nothing.
    """
    return batch


# def get_dataloaders(cfg: cfg, df, fold=0):
#     train_dataset, valid_dataset = get_datasets(cfg, df, fold)

#     if len(train_dataset) > 0:
#         train_loader = DataLoader(train_dataset, batch_size=cfg.train.bs,
#                                 num_workers=2, collate_fn=trivial_batch_collator, 
#                                 shuffle=True, pin_memory=True, drop_last=False)
#     else:
#         train_loader = None
    
#     if len(valid_dataset) > 0:
#         valid_loader = DataLoader(valid_dataset, batch_size=cfg.valid.bs,
#                                 num_workers=2, collate_fn=trivial_batch_collator, 
#                                 shuffle=False, pin_memory=True)
#     else:
#         valid_loader = None

#     return train_loader, valid_loader








def build_loader(
    dataset: Dataset,
    *,
    batch_size: int = 1,
    num_workers: int = 0,
    collate_fn: Optional[Callable[[List[Any]], Any]] = None,
) -> DataLoader:
    """
    Similar to `build_detection_train_loader`, with default batch size = 1,
    and sampler = :class:`InferenceSampler`. This sampler coordinates all workers
    to produce the exact set of all samples.

    Args:
        dataset: nn.Dataset class,
            or a pytorch dataset. They can be obtained
            by using :func:`DatasetCatalog.get`.
        batch_size: the batch size of the data loader to be created.
            Default to 1 image per worker since this is the standard when reporting
            inference time in papers.
        num_workers: number of parallel data loading workers
        collate_fn: same as the argument of `torch.utils.data.DataLoader`.
            Defaults to do no collation and return a list of data.

    Returns:
        DataLoader: a torch DataLoader, that loads the given detection
        dataset, with test-time transformation and batching.
    """
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        collate_fn=trivial_batch_collator if collate_fn is None else collate_fn,
    )




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
    
