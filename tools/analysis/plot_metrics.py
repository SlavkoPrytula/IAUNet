import matplotlib.pyplot as plt
import pandas as pd


def preprocess_columns(dataframe):
    stripped_columns = {col: col.strip() for col in dataframe.columns}
    return dataframe.rename(columns=stripped_columns)


def plot_metrics(csv_dict, metric_name, xlabel='Epochs', ylabel='Metric Value'):

    plt.figure(figsize=(10, 6))
    
    colors = ['#FFB000', '#DC267F', '#648FFF', '#FE6100', '#785EF0', '#FFB5A7', '#80CBE5', '#1D3557', '#A8DADC', '#457B9D']
    
    for idx, (name, path) in enumerate(csv_dict.items()):
        df = pd.read_csv(path)
        df = preprocess_columns(df)

        if metric_name in df.columns:
            plt.plot(df['epoch'], df[metric_name], linewidth=4,
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
    plt.savefig("./tools/analysis/outputs/run_analysis.jpg")


csv_dict = {
    'Run 1': "runs/[resnet_iaunet_multitask]/[truncated_decoder-iadecoder_ml]/[ResNet]/[LiveCellCrop]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[InstanceHead-v1.1]/[job=51958106]-[2024-08-27 00:14:43]/results.csv",
    'Run 2': "runs/[resnet_iaunet_multitask]/[truncated_decoder-iadecoder_ml]/[ResNet]/[LiveCellCrop]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[InstanceHead-v1.1]/[job=51959650]-[2024-08-27 15:00:50]/results.csv",
    'Run 3': "runs/[resnet_iaunet_multitask_ml]/[truncated_decoder-iadecoder_ml]/[ResNet]/[LiveCellCrop]/[softmax_iam]/[kernel_dim=256]-[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[InstanceHead-v2.2-two-way-attn]/[job=51967871]-[2024-08-28 10:13:47]/results.csv"
}

plot_metrics(csv_dict, 'loss_valid')

# epoch, mAP@0.5:0.95,      mAP@0.5,     mAP@0.75,   mAP(s)@0.5,   mAP(m)@0.5,   mAP(l)@0.5,     mean_IoU,   loss_valid,loss_ce_valid,loss_bce_masks_valid,loss_dice_masks_valid,   loss_train,loss_ce_train,loss_bce_masks_train,loss_dice_masks_train
