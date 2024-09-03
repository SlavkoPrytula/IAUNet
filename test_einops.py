# # import sys
# # sys.path.append("./")

# # from configs import cfg
# # from utils.registry import VISUALIZERS
# # from visualizations.visualizers import *

# # cfg.save_dir = ""
# # visualizer = VISUALIZERS.build(cfg.visualizer)
# # print(visualizer.visualizers.inst_type)
# # # visualizer.on_train_epoch_end(cfg=cfg, epoch=1, output="some_output")
# # # print(visualizer.output)
# # # print(visualizer.visualizers["iam_visualizer"].output)



# # import torch
# # import torch.nn as nn
# # from torch.nn import functional as F

# # from models.seg.nn.blocks import CrossAttentionLayer
    

# # layer = CrossAttentionLayer(20, 1)
# # x = torch.rand(2, 5, 20)
# # y = torch.rand(2, 5, 20)
# # z = layer(x, y)
# # print(z.shape)



# import time
# import torch
# import einops
# from tqdm import tqdm

# # Define the dimensions
# B = 16  # batch size
# Q = 100  # number of queries
# C = 256  # channels
# H = 128  # height
# W = 128  # width

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)

# # Create random tensors on CUDA
# bqc = torch.randn(B, Q, C).to(device)
# bchw = torch.randn(B, C, H, W).to(device)

# # Function to test einsum
# def test_einsum():
#     # return einops.einsum(bqc, bchw, "b q c, b c h w -> b q h w")
    
#     return bchw.view(B, C, -1)

# # Function to test bmm
# def test_bmm():
#     # bchw_flat = bchw.view(B, C, -1)  # Flatten the spatial dimensions
#     # result_flat = torch.bmm(bqc, bchw_flat)  # Perform batch matrix multiplication
#     # return result_flat.view(B, Q, H, W)  # Reshape back to the original spatial dimensions
    
#     return bchw.flatten(2)

# # Number of iterations
# iterations = 4000

# # Test einsum
# einsum_times = []
# for _ in tqdm(range(iterations)):
#     start_time = time.time()
#     test_einsum()
#     torch.cuda.synchronize()  # Ensure all CUDA ops are finished
#     end_time = time.time()
#     einsum_times.append(end_time - start_time)

# # Test bmm
# bmm_times = []
# for _ in tqdm(range(iterations)):
#     start_time = time.time()
#     test_bmm()
#     torch.cuda.synchronize()  # Ensure all CUDA ops are finished
#     end_time = time.time()
#     bmm_times.append(end_time - start_time)

# # Calculate average and best times
# einsum_avg_time = sum(einsum_times) / iterations
# einsum_best_time = min(einsum_times)

# bmm_avg_time = sum(bmm_times) / iterations
# bmm_best_time = min(bmm_times)

# # Print the results
# print(f"Einsum average time: {einsum_avg_time * 1000:.6f} ms")
# print(f"Einsum best time: {einsum_best_time * 1000:.6f} ms")
# print(f"BMM average time: {bmm_avg_time * 1000:.6f} ms")
# print(f"BMM best time: {bmm_best_time * 1000:.6f} ms")




import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

# -----
logits = torch.tensor([[2.0, 1.0, 0.1]])
true_label = torch.tensor([0])

loss_full = criterion(logits, true_label)

# -----
logits_group1 = torch.tensor([[2.0, 1.0]])
logits_group2 = torch.tensor([[0.1]])
label_group1 = torch.tensor([0])
label_group2 = torch.tensor([0])

loss_group1 = criterion(logits_group1, label_group1)
loss_group2 = criterion(logits_group2, label_group2)

total_loss_split = loss_group1 + loss_group2

# -----
print("Full loss:", loss_full.item())
print("Loss Group 1:", loss_group1.item())
print("Loss Group 2:", loss_group2.item())
print("Total Split Loss:", total_loss_split.item())
print()


import torch
import torch.nn as nn

criterion = nn.CrossEntropyLoss()

logits = torch.tensor([[2.0, 1.0, 0.1]])
true_label = torch.tensor([0])

loss_full = criterion(logits, true_label)

logits_group1 = torch.tensor([[2.0, 1.0, -10]])
logits_group2 = torch.tensor([[-10, -10, 0.1]])
label_group1 = torch.tensor([0])
label_group2 = torch.tensor([0])

loss_group1 = criterion(logits_group1, label_group1)
loss_group2 = criterion(logits_group2, label_group2)

total_loss_split = loss_group1 + loss_group2

print("Full loss:", loss_full.item())
print("Loss Group 1:", loss_group1.item())
print("Loss Group 2:", loss_group2.item())
print("Total Split Loss:", total_loss_split.item())
print()


import torch.nn.functional as F

logits = torch.tensor([[2.0, 1.0, 0.1]])
probs = F.softmax(logits, dim=1)
print(f'full probs: {probs}')


logits_group1 = torch.tensor([[2.0, 1.0, -10]])
logits_group2 = torch.tensor([[-10, -10, 0.1]])
probs1 = F.softmax(logits_group1, dim=1)
probs2 = F.softmax(logits_group2, dim=1)
print(f'probs1: {probs1}')
print(f'probs2: {probs2}')



