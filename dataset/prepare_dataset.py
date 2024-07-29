from sklearn.model_selection import StratifiedKFold, KFold, StratifiedGroupKFold


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
