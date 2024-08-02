import time
from typing import List
import numpy as np
import torch

from .train import train_one_epoch
from .valid import valid_one_epoch

from configs import cfg


def run_training(
        cfg: cfg, 
        model, 
        criterion, 
        train_dataloader, 
        valid_dataloader, 
        optimizer, 
        scheduler,
        evaluators,
        callbacks,
        accelerator,
        logger,
        ):
    
    if accelerator == 'gpu' and torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(device)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU\n")

    csv = cfg.run.save_dir / 'results.csv'

    start = time.time()
    best_loss = np.inf

    num_epochs = cfg.trainer.max_epochs + 1
    for epoch in range(0, num_epochs):
        logger.info(f'Epoch {epoch}/{num_epochs}')

        results = {}
        results_train = train_one_epoch(cfg, model, criterion=criterion, 
                                        optimizer=optimizer, 
                                        scheduler=scheduler,
                                        dataloader=train_dataloader,
                                        device=device, 
                                        epoch=epoch,
                                        callbacks=callbacks,
                                        logger=logger,
                                        )

        if epoch % cfg.trainer.check_val_every_n_epoch == 0:
            results_valid = valid_one_epoch(cfg, model, criterion=criterion, 
                                            optimizer=optimizer, 
                                            scheduler=scheduler,
                                            dataloader=valid_dataloader,
                                            device=device, 
                                            epoch=epoch,
                                            evaluators=evaluators,
                                            callbacks=callbacks,
                                            logger=logger,
                                            )

            results.update(results_valid)
            results.update(results_train)

            val_loss = results["loss_valid"]
            if val_loss <= best_loss:
                logger.info(f"Valid Loss Improved ({best_loss:0.4f} ---> {val_loss:0.4f})")
                best_loss = val_loss
                
                # saving best model.
                if cfg.model.save_checkpoint:
                    torch.save(model.state_dict(), cfg.run.save_dir / 'checkpoints/best.pth')
                    print()

            metrics = list(results.keys())
            vals = list(results.values())

            s = '' if csv.exists() else (('%13s,' * (len(metrics) + 1) % tuple(['epoch'] + metrics)).rstrip(',') + '\n')  # header
            with open(csv, 'a') as f:
                f.write(s + ('%13.5g,' * (len(metrics) + 1) % tuple([epoch] + vals)).rstrip(',') + '\n')

        # saving last checkpoint.
        if cfg.model.save_checkpoint:
            torch.save(model.state_dict(), cfg.run.save_dir / 'checkpoints/last.pth')

    end = time.time()
    time_elapsed = end - start
    logger.info('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))

    return model
