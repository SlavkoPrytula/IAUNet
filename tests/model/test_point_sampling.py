import sys
sys.path.append("./")

from models.losses.loss import get_uncertain_point_coords_with_randomness, calculate_uncertainty
from models.losses.matcher import point_sample
import torch
import numpy as np

# --------------------
# 0. calculate_uncertainty
# uncertainty_func will calculate the uncertainties on a map raw logits
# as input the function expects a tensor of shape (N, 1, P)
# where:
# - P is the number of points 
# - N is the number of samples
# calculate_uncertainty() estimates the uncertainty scores 
# as the L1 difference between 0.0 and the logit values

# returns uncertainties as a tensor of shape (N, 1, P), where the most unceratain points 
# have the highest values scores

# --------------------
# 1. point_sample
# Given a flow-field grid computes the output using the input values and the pixel locations from the grip
# Basically the function samples the input values at the pixel locations defined by the grid by 
# using the bilinear interpoltation (the ouput at each grid location is computed as a 
# weighted average of the four nearest input values - each weight is proportional to the distance)
# :note: the function expects normalized grid coordinates in the range (0-1)
# but internally we renormalize these to the (-1, 1) range and feed it as the input to the F.grid_sample() function
# inputs:
# - input: F.grid_sample() expects an 4-D (or 5-D) input tensor (N, C, H, W)
# - point_coords: a tensor of shape (N, P, 2), where the tensor is a grid if uniformly sampled points in (0-1) range
# returns a tensor of shape (N, C, P)

# --------------------
# 2.get_uncertain_point_coords_with_randomness
# samples the points in a [0, 1] x [0, 1] grid space based on the uncertainty scores
# the function expects a tensor of shape (N, 1, H, W) as input
# inputs:
# - coarse_logits: tensor of shape (N, 1, H, W)
# - uncertainty_function: a function that calculates the uncertainty scores
# - num_points: number of points to sample
# - oversampling_ratio: ratio of points to sample in addition to the num_points (???)
# - importance_sampling_ratio: (???)

# usually we define the num_points to be 112 * 112 = 12544
# the oversampling_ration is by default set to 3.0 meaning we will sample 3 * 12544 = 37632 points
# then the function creates a grid of points in the [0, 1] x [0, 1] space | using the torch.rand(N, num_points, 2) function
# - for each of N samples, it will create such a grid of points 

# then we feed this grid of normalized coordinates and the input coarse_logits to the point_sample() function
# - it will sample the logits at the grid points 

# then on the sampled logits we apply the unceratinty_function() to get the unceratainty scores for each point
# after we need sample the uncrtain points and random points and concatenate them to get the final point coordinates
# of shape (N, num_uncertain_points + num_random_points, 2) = (N, P, 2)





point_coords = torch.rand(5, 100, 2)

point_uncertainties = torch.rand(5, 1, 100)  
num_uncertain_points = int(100 * 0.75)
idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
# ids: (N, num_uncertain_points)

shift = 100 * np.arange(5)
idx += shift[:, None]
# idx: (N, num_uncertain_points) - for each sample N idx we add a shift value
# in order ot correstly index the uncertain points for each sample 

point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(5, num_uncertain_points, 2)
print(point_coords.shape)

num_random_points = 100 - num_uncertain_points
rand_points = torch.rand(5, num_random_points, 2)
print(rand_points.shape)

point_coords = torch.cat([
    point_coords,
    rand_points
], dim=1)

print(point_coords.shape)


