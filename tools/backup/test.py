if __name__ == "__main__":
    # =================================
    # ============ Dataset ============
    # =================================

    # from dataset.datasets import df
    # from dataset.datasets.brightfiled_v2 import Brightfield_Dataset

    # from utils.normalize import normalize
    # from utils.augmentations import train_transforms
    # from utils.visualise import visualize
    # from utils.utils import flatten_mask
    # from configs import cfg
    
    # train_dataset = Brightfield_Dataset(
    #     df=df,
    #     run_type='train',
    #     img_size=cfg.train.size,
    #     normalization=normalize,
    #     transform=train_transforms(cfg)
    # )
    # bf, pc, cyto_mask, nuc_mask, cond_mask, dx_grad, dy_grad = train_dataset[4]
    # print(pc.shape)
    # print(cyto_mask.shape)
    # print(cyto_mask.min(), cyto_mask.max())
    # print(dx_grad.min(), dx_grad.max())

    # visualize(
    #     [20, 8],
    #     bf_lo=bf[0, ...],
    #     bf_hi=bf[1, ...],
    #     pc=pc[0, ...],
    #     cyto_mask=flatten_mask(cyto_mask.cpu().detach().numpy(), axis=0)[0, ...],
    #     nuc_mask=flatten_mask(nuc_mask.cpu().detach().numpy(), axis=0)[0, ...],
    # #     mask_sample=cyto_mask[0, ...],
    # #     cond_mask=cond_mask[0, ...]
    #     dx_grad=dx_grad[0, ...],
    #     dy_grad=dy_grad[0, ...],
    #     path='./dataset.jpg'
    # )

    # =================================
    
    # =================================
    # ============= Model =============
    # =================================

    # import torch
    # from configs import cfg
    # from models.build_model import build_model
    # import gc

    # gc.collect()
    # torch.cuda.empty_cache()
    # model = build_model(cfg=cfg)

    # x = torch.randn(1, 2, 512, 512).to(cfg.device)
    # out = model(x)

    # print(out[2].shape)

    # =================================

    
    # =================================
    # ======== InstanceBranch =========
    # =================================

    # import torch
    # from configs import cfg
    # from models.seg.heads.instance_head import InstanceBranch
    
    # instance_decoder = InstanceBranch(32, 128, num_masks=10).to(cfg.device)
    # x = torch.randn(2, 32, 64, 64).to(cfg.device)

    # logits, kernel, scores, iam = instance_decoder(x)

    # print(logits.shape)   # class pred
    # print(kernel.shape)   # mask kernel
    # print(scores.shape)   # scores for each mask
    # print(iam.shape)      # logit instance mask proposals


    # # =================================

    
    # # =================================
    # # ========== Matrix NMS ===========
    # # =================================

    # import torch
    # import numpy as np
    # import matplotlib.pyplot as plt

    # from utils.visualise import visualize
    # from utils.utils import flatten_mask

    # from utils.post_processing.matrix_nms import mask_matrix_nms
    

    # masks = np.zeros((6, 20, 20))
    # masks[0, :5, :5] = 1
    # masks[3, 2:6, :5] = 1
    # masks[2, 6:12, 7:10] = 1

    
    # masks = torch.tensor(masks)
    # labels = torch.tensor([1., 1., 1., 1., 1., 1.])
    # scores = torch.tensor([1., 0.7, 1., 0.5, 1., 0.5])

    # masks = masks.type(torch.uint8)

    # sum_masks = masks.sum((1, 2)).float()
    # keep = sum_masks.nonzero()[:, 0]

    # masks = masks[keep]
    # scores = scores[keep]
    # labels = labels[keep]

    # print(masks.shape)
    # print(scores.shape)
    # print(labels.shape)


    # visualize(mask=flatten_mask(masks.cpu().detach().numpy(), 0)[0, ...], path='martix_nms_before.jpg')

    # scores, labels, masks, keep_inds = mask_matrix_nms(masks=masks, labels=labels, scores=scores, filter_thr=0.5)
    # print(scores)
    # print(labels)
    # print(keep_inds)



    # visualize(mask=flatten_mask(masks.cpu().detach().numpy(), 0)[0, ...], path='martix_nms_after.jpg')



    # =================================

    
    # =================================
    # ======= Similarity Matrix =======
    # =================================

    # import torch
    # import numpy as np
    # import matplotlib.pyplot as plt

    # from utils.visualise import visualize, visualize_grid_v2
    # from utils.utils import flatten_mask

    # from utils.post_processing.matrix_nms import mask_matrix_nms


    # masks = np.zeros((25, 20, 20))
    # masks[0, :5, :5] = 1
    # masks[3, 2:6, :5] = 1
    # masks[2, 6:12, 7:10] = 1

    # masks[4, 16:20, 16:20] = 1
    # masks[5, 17:19, 17:19] = 1
    # masks[6, 1:15, 10:15] = 1
    # masks[7, 6:13, 12:15] = 1

    # masks[17, :5, :5] = 1
    # masks[17, 7:16, 10:14] = 0.5

    
    # masks = torch.tensor(masks)
    # labels = torch.tensor([1., 1., 1., 1., 1., 1.])
    # # scores = torch.tensor([1., 0.7, 1., 0.5, 1., 0.5])
    # scores = torch.rand(25)

    # masks = masks.type(torch.float64)

    # sum_masks = masks.sum((1, 2)).float()
    # keep = sum_masks.nonzero()[:, 0]

    # masks = masks[keep]
    # scores = scores[keep]

    # print(masks.shape)
    # print(scores.shape)
    # print(labels.shape)

    # visualize_grid_v2(masks=masks.cpu().detach().numpy(), titles=scores.cpu().detach().numpy(), ncols=3, path='test.jpg')


    # def pairwise_jaccard(masks):
    #     n_masks, height, width = masks.shape

    #     # Flatten masks and compute pairwise intersections and unions
    #     masks_flat = masks.reshape(n_masks, -1)
    #     intersections = np.dot(masks_flat, masks_flat.T)
    #     unions = np.sum(masks_flat, axis=1)[:, None] + np.sum(masks_flat, axis=1) - intersections

    #     # Compute Jaccard index
    #     jaccard = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions!=0)

    #     return jaccard
    

    # def dice_similarity(masks):
    #     n = masks.shape[0]
    #     sim_mat = np.zeros((n, n))
    #     for i in range(n):
    #         for j in range(i, n):
    #             intersection = np.logical_and(masks[i], masks[j]).sum()
    #             union = masks[i].sum() + masks[j].sum()
    #             sim_mat[i, j] = 2 * intersection / union
    #             sim_mat[j, i] = sim_mat[i, j]
    #     return sim_mat
    

    # def dice_score(inputs):
    #     N, H, W = inputs.shape
    #     inputs = inputs.view(N, -1)
    #     numerator = 2 * torch.matmul(inputs, inputs.t())
    #     denominator = (inputs * inputs).sum(-1)[:, None] + (inputs * inputs).sum(-1)
    #     score = numerator / (denominator + 1e-4)
    #     return score

    
    # # sim_matrix = pairwise_jaccard(masks.cpu().detach().numpy())
    # sim_matrix = dice_score(masks)
    # print(sim_matrix)



    # def similarity_graph(A, thr=0.25):
    #     N = A.shape[0]
    #     groups = []
    #     visited = set()
    #     for i in range(N):
    #         if i not in visited:
    #             group = [i]
    #             visited.add(i)
    #             for j in range(i+1, N):
    #                 if A[i,j] >= thr:
    #                     group.append(j)
    #                     visited.add(j)
    #             if len(group) > 0:
    #                 groups.append(tuple(group))
    #     return groups
    
    # def remove_duplicate_nodes(G):
    #     visited_nodes = []
    #     for g in G:
    #         visited_nodes.extend(list(g))

    #     duplicate_nodes = set([x for x in visited_nodes if visited_nodes.count(x) > 1])

    #     filtered_nodes = []
    #     for g in G:
    #         new_g = tuple([x for x in g if x not in duplicate_nodes])
    #         filtered_nodes.append(new_g)

    #     return filtered_nodes

    # sim_G = similarity_graph(sim_matrix, thr=0.2)
    # print(sim_G)

    # sim_G = remove_duplicate_nodes(sim_G)
    # print(sim_G)


    # def mask_fusion(masks, G):
    #     def _fuse(masks):
    #         # return np.mean(masks, axis=0)

    #         masks = np.sum(masks, axis=0)
    #         masks[masks > 1] = 1
    #         return masks
        
    #     fused_masks = []
    #     for g in G:
    #         g = np.array(g)
    #         selected_masks = masks[g]
    #         merged_mask = _fuse(selected_masks)
    #         fused_masks.append(merged_mask)

    #     fused_masks = np.stack(fused_masks, axis=0)
    #     return fused_masks
    
    # fused_masks = mask_fusion(masks.cpu().detach().numpy(), sim_G)
    # print(fused_masks.shape)
    # visualize_grid_v2(masks=fused_masks, ncols=3, path='test_fused.jpg')




# [[1.         0.         0.5        0.         0.         0.         0.         0.625     ]
#  [0.         1.         0.         0.         0.         0.         0.         0.09433962]
#  [0.5        0.         1.         0.         0.         0.         0.         0.33333333]
#  [0.         0.         0.         1.         0.25       0.         0.         0.        ]
#  [0.         0.         0.         0.25       1.         0.         0.         0.        ]
#  [0.         0.         0.         0.         0.         1.         0.3        0.1       ]
#  [0.         0.         0.         0.         0.         0.3        1.         0.        ]
#  [0.625      0.09433962 0.33333333 0.         0.         0.1        0.         1.        ]]


# thr: 0.0
# --------
# 0: 2, 7
# 1: 7
# 2: 0, 7 
# 3: 4
# 4: 3
# 5: 6, 7
# 6: 5
# 7: 0, 1, 2, 5


# thr: 0.25
# --------
# 0: 2, 7
# 1: 
# 2: 0, 7 
# 3: 4
# 4: 3
# 5: 6
# 6: 5
# 7: 0, 2

# (1), (3, 4), (5, 6), (0, 2, 7) 





    # =================================

    
    # =================================
    # ============== CFG ==============
    # =================================

    from configs import cfg
    print(cfg.model.__dict__)