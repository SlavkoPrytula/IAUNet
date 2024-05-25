from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold
# from .datasets.brightfiled import Brightfield_Dataset
# from .datasets.brightfiled_nuc import Brightfield_Nuc_Dataset
from .datasets.rectangle import Rectangle_Dataset


def get_folds(cfg, df):
    # Satisfied KFold Split - by [types]
    if cfg.train.n_folds == 1:
        df['fold'] = 0
    else:
        skf = StratifiedGroupKFold(n_splits=cfg.train.n_folds, shuffle=True, random_state=cfg.seed)
        for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['cell_line'], groups=df['id'])):
            df.loc[val_idx, 'fold'] = fold

    return df


def build_dataset(name: str):
    datasets = {
        # 'brightfiled': Brightfield_Dataset,
        # 'brightfiled_nuc': Brightfield_Nuc_Dataset,
        'rectangle': Rectangle_Dataset
    }

    dataset = datasets[name]

    return dataset
