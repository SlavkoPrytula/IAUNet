import torch
import torch.multiprocessing as mp
from torch import nn
import os
import sys
sys.path.append("./")

from utils.dist.comm import setup
from utils.dist import init_dist, collect_results, get_rank, get_world_size, is_main_process, broadcast_object_list

def _to_cpu(data):
    if isinstance(data, torch.Tensor):
        return data.cpu()
    elif isinstance(data, list):
        return [_to_cpu(d) for d in data]
    elif isinstance(data, dict):
        return {k: _to_cpu(v) for k, v in data.items()}
    else:
        return data

# Simple model for testing
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class SimpleMetric:
    def __init__(self):
        self.results = []
        self.collect_device = 'cpu'
        self.collect_dir = None

    def process(self, outputs):
        self.results.append(outputs)

    def compute_metrics(self, results):
        # Aggregate results across all ranks
        aggregated_results = torch.cat(results, dim=0)
        mean_metric = aggregated_results.mean().item()
        return {'mean_metric': mean_metric}

    def evaluate(self, size):
        if len(self.results) == 0:
            print('No results to evaluate.')

        print(f"[Rank {get_rank()}] Results before collection: {self.results}")

        # Collect results across all ranks
        if self.collect_device == 'cpu':
            results = collect_results(
                self.results,
                size,
                self.collect_device,
                tmpdir=self.collect_dir)
        else:
            results = collect_results(self.results, size, self.collect_device)

        print(f"[Rank {get_rank()}] Results after collection: {results}")

        metrics = None
        if is_main_process():
            # Cast all tensors in results list to CPU
            results = _to_cpu(results)
            metrics = self.compute_metrics(results)
            print(f"Rank {get_rank()} - Computed Metrics: {metrics}")
        
        # Broadcast the computed metrics to all processes
        metrics_list = [metrics]
        broadcast_object_list(metrics_list, src=0)

        # Reset the results list
        self.results.clear()
        return metrics_list[0]


def train(rank, world_size):
    setup(rank, world_size)

    # Initialize model and data
    model = SimpleModel().to(rank)
    data = torch.randn(5, 10).to(rank)
    
    # Forward pass
    outputs = model(data)

    # Initialize and process metric
    metric = SimpleMetric()
    metric.process(outputs)

    # Evaluate metric and check synchronization
    metric.evaluate(size=5 * world_size)

    # Clean up
    torch.distributed.destroy_process_group()


def main():
    world_size = 2  # Number of GPUs
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
