import matplotlib.pyplot as plt
import pandas as pd
from os import makedirs


def preprocess_columns(dataframe):
    stripped_columns = {col: col.strip() for col in dataframe.columns}
    return dataframe.rename(columns=stripped_columns)


def plot_metrics(csv_dict, metric_name, xlabel='Epochs', ylabel='Metric Value'):
    save_path = './tools/analysis/outputs'
    makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(10, 6), dpi=300)
    
    colors = ['#FFB000', '#DC267F', '#648FFF', '#FE6100', '#785EF0', '#FFB5A7', '#80CBE5', '#1D3557', '#A8DADC', '#457B9D']
    
    for idx, (name, path) in enumerate(csv_dict.items()):
        df = pd.read_csv(path)
        df = preprocess_columns(df)

        if metric_name in df.columns:
            # Apply rolling average
            rolling_metric = df[metric_name].rolling(window=1, min_periods=1).mean()
            plt.plot(df['epoch'], rolling_metric, linewidth=3,
                     label=name, color=colors[idx % len(colors)])
        else:
            print(f"Metric '{metric_name}' not found in {path}.")
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.ylim([0.6, 0.9])
    plt.title(f'{metric_name}')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f"{save_path}/run_analysis.jpg")


csv_dict = {
    'v1': "runs/benchmarks/[Revvity_25]/[iaunet-r101]/[iadecoder_ml_fpn]/[InstanceHead-v3.t-testing]/[job=8434992]-[2024-11-13 01:25:25]/results.csv", 
    'v2': "runs/benchmarks/[Revvity_25]/[iaunet-r50]/[iadecoder_ml_fpn]/[InstanceHead-v2.2-two-way-attn]/[job=8693932]-[2024-12-10 16:08:44]/results.csv", 
    'v2 + ca': "runs/experiments/[Revvity_25]/[iaunet-r50]/[iadecoder_ml_fpn_ia_queries]/[job=8708274]-[2024-12-11 13:57:51]/results.csv", 
}

plot_metrics(csv_dict, 'mAP@0.5:0.95')

# epoch, mAP@0.5:0.95,      mAP@0.5,     mAP@0.75,   mAP(s)@0.5,   mAP(m)@0.5,   mAP(l)@0.5,     mean_IoU,   loss_valid,loss_ce_valid,loss_bce_masks_valid,loss_dice_masks_valid,   loss_train,loss_ce_train,loss_bce_masks_train,loss_dice_masks_train
