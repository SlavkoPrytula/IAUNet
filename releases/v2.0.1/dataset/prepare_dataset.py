from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
# from .datasets.original_plus_synthetic_brightfield import OriginalPlusSyntheticBrightfield
# from .datasets.synthetic_brightfield import SyntheticBrightfield_Dataset
# from .datasets.brightfiled_nuc import Brightfield_Nuc_Dataset
# from .datasets.brightfiled import Brightfield_Dataset
# from .datasets.rectangle import Rectangle_Dataset


def get_folds(cfg, df):
    if 'fold' in df.columns:
        print('WARNING: using predifined folds')
        # uses user-predefines folds for train/valids split
        return df

    # Satisfied KFold Split - by [types]
    if cfg.train.n_folds == 1:
        df['fold'] = 0
    else:
        skf = StratifiedGroupKFold(n_splits=cfg.train.n_folds, shuffle=True, random_state=cfg.seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['cell_line'], groups=df['id'])):
            df.loc[val_idx, 'fold'] = fold

    return df


# def build_dataset(name: str):
#     datasets = {
#         # 'original_plus_synthetic_brightfield': OriginalPlusSyntheticBrightfield,
#         # 'synthetic_brightfield': SyntheticBrightfield_Dataset
#         # 'brightfield_nuc': Brightfield_Nuc_Dataset,
#         # 'brightfield': Brightfield_Dataset,
#         'rectangle': Rectangle_Dataset
#     }

#     dataset = datasets[name]

#     return dataset


