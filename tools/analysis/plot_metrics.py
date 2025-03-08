import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from os import makedirs


def preprocess_columns(dataframe):
    stripped_columns = {col: col.strip() for col in dataframe.columns}
    return dataframe.rename(columns=stripped_columns)


def plot_metrics(csv_dict, metric_name, xlabel='Epochs', ylabel='Metric Value'):
    save_path = './tools/analysis/outputs'
    makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(6, 6), dpi=300)
    
    colors = ['#FFB000', '#DC267F', '#648FFF', '#FE6100', '#785EF0', '#FFB5A7', '#80CBE5', '#1D3557', '#A8DADC', '#457B9D']
    
    for idx, (name, path) in enumerate(csv_dict.items()):
        df = pd.read_csv(path)
        df = preprocess_columns(df)

        if metric_name in df.columns:
            # Apply rolling average
            rolling_metric = df[metric_name].rolling(window=1, min_periods=1).mean()
            plt.plot(df['epoch'], rolling_metric, linewidth=2, 
                     label=name, color=colors[idx % len(colors)])
        else:
            print(f"Metric '{metric_name}' not found in {path}.")

    plt.gca().yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim([0, 1])
    plt.title(f'{metric_name}')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"{save_path}/run_analysis.jpg")


# csv_dict = {
#     'v8 (clip_grad=0.01)': "runs/experiments_v2/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn]/[experimental]/[clip_grad_norm]/[job=9585495]-[2025-02-18 11:58:57]/results.csv", 
#     'v2': "runs/experiments_v2/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn]/[job=9541227]-[2025-02-15 13:50:42]/results.csv", 
# }

# csv_dict = {
#     'v11 (dec_layers=3 + ds)': "runs/experiments_v2/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn]/[experimental]/[deep_supervision]/[job=9605998]-[2025-02-19 18:44:47]/results.csv", 
#     'v2': "runs/experiments_v2/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn]/[job=9541227]-[2025-02-15 13:50:42]/results.csv", 
# }

csv_dict = {
    'v11 (dec_layers=3 + ds)': "runs/experiments_v2/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn]/[experimental]/[deep_supervision]/[job=9605998]-[2025-02-19 18:44:47]/results.csv", 
    'v5  (dec_layers=3)': "runs/experiments_v2/[LiveCellCrop]/[iaunet-r50]/[iadecoder_ml_fpn]/[experimental]/[mask_decoupling]/[job=9562134]-[2025-02-16 19:33:24]/results.csv", 
}

plot_metrics(csv_dict, 'mAP@0.5')

# epoch, mAP@0.5:0.95,      mAP@0.5,     mAP@0.75,   mAP(s)@0.5,   mAP(m)@0.5,   mAP(l)@0.5,     mean_IoU,   loss_valid,loss_ce_valid,loss_bce_masks_valid,loss_dice_masks_valid,   loss_train,loss_ce_train,loss_bce_masks_train,loss_dice_masks_train
