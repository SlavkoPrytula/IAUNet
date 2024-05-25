import gc
import time
import copy
from collections import defaultdict

import numpy as np
import torch

from .train import train_one_epoch
from .valid import valid_one_epoch


def run_training(cfg, model, losses, optimizer, train_loader, valid_loader, 
                 fold_i, scheduler=None, num_epochs=1, run=None):
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))

    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss_wts = copy.deepcopy(model.state_dict())
    best_mse = np.inf
    best_loss = np.inf
    best_epoch = -1
    history = defaultdict(list)

    # To automatically log gradients
    if run:
        run.watch(model, log_freq=100)

    for epoch in range(0, num_epochs):
        gc.collect()

        print(f'Epoch {epoch}/{num_epochs}')

        results = train_one_epoch(cfg, model, losses, optimizer, scheduler,
                                     dataloader=train_loader,
                                     device=cfg.device, 
                                     epoch=epoch)

        results = valid_one_epoch(cfg, model, losses, optimizer,
                                    dataloader=valid_loader,
                                    device=cfg.device,
                                    epoch=epoch)


        # train_loss = results["train_loss"]
        val_loss = results["loss_valid"]
        
        # history['Train Loss'].append(train_loss)
        # history['Valid Loss'].append(val_loss)

        if val_loss <= best_loss:
            print(f"Valid Loss Improved ({best_loss:0.4f} ---> {val_loss:0.4f})")
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(), cfg.save_dir / 'checkpoints/best.pth')
            print()


        if epoch % 10 == 0:
            metrics = list(results.keys())
            vals = list(results.values())

            s = '' if cfg.csv.exists() else (('%13s,' * (len(metrics) + 1) % tuple(['epoch'] + metrics)).rstrip(',') + '\n')  # header
            with open(cfg.csv, 'a') as f:
                f.write(s + ('%13.5g,' * (len(metrics) + 1) % tuple([epoch] + vals)).rstrip(',') + '\n')


    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model
