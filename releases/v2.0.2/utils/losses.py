# import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

# JaccardLoss = smp.losses.JaccardLoss(mode='binary')
# DiceLoss = smp.losses.DiceLoss(mode='binary')
# BCELoss = smp.losses.SoftBCEWithLogitsLoss()
# LovaszLoss = smp.losses.LovaszLoss(mode='binary')
# TverskyLoss = smp.losses.TverskyLoss(mode='binary', log_loss=False)


def criterion(y_pred, y_true):
    # return 0.2 * BCELoss(y_pred, y_true) + 0.8 * DiceLoss(y_pred, y_true)
    return F.binary_cross_entropy_with_logits(y_pred, y_true, reduction='mean')


def compute_mask_iou(inputs, targets):
    inputs = inputs.sigmoid()
    # thresholding
    binarized_inputs = (inputs >= 0.4).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def dice_score(inputs, targets):
    inputs = inputs.sigmoid()
    numerator = 2 * torch.matmul(inputs, targets.t())
    denominator = (inputs * inputs).sum(-1)[:, None] + (targets * targets).sum(-1)
    score = numerator / (denominator + 1e-4)
    return score


def dice_loss(inputs, targets, reduction='sum'):
    inputs = inputs.sigmoid()
    assert inputs.shape == targets.shape

    numerator = 2 * (inputs * targets).sum(1)
    denominator = (inputs * inputs).sum(-1) + (targets * targets).sum(-1)
    loss = 1 - (numerator) / (denominator + 1e-4)

    if reduction == 'none':
        return loss
    return loss.sum()



def dice_loss_detr(inputs, targets, num_masks):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


# def dice_loss(inputs, targets, weight=None, reduction='sum'):
#     inputs = inputs.sigmoid()
#     assert inputs.shape == targets.shape
    
#     if weight is not None:
#         numerator = 2 * (inputs * targets * weight).sum(1)  # sum in [HW] dim
#         denominator = ((inputs * inputs) * weight).sum(-1) + ((targets * targets) * weight).sum(-1)
#         loss = 1 - (numerator) / (denominator + 1e-4)
#     else:
#         numerator = 2 * (inputs * targets).sum(1)
#         denominator = (inputs * inputs).sum(-1) + (targets * targets).sum(-1)
#         loss = 1 - (numerator) / (denominator + 1e-4)

#     if reduction == 'none':
#         return loss
#     return loss.sum()



def DiceLossFP(gt, pred, alpha=0.5, beta=1):
    """Compute the modified Dice loss between ground truth and predicted binary masks
       with a higher weight for false positives."""
    smooth = 1e-6

    # Compute weight for each ground truth mask
    w = 1 / (torch.sum(gt, dim=(1,2)) + smooth)

    # Compute intersection and union between masks
    intersection = torch.sum(gt * pred, dim=(1,2))
    union = torch.sum(gt + pred, dim=(1,2))

    # Compute Dice coefficient
    dice = 2 * torch.sum(w * intersection) / (torch.sum(w * union) + smooth)

    # Compute weight for false positives
    fp_weight = beta * torch.sum((1 - gt) * pred, dim=(1,2)) / (torch.sum(pred, dim=(1,2)) + smooth)

    # Compute weighted intersection and union
    w_intersection = intersection + alpha * fp_weight
    w_union = union + beta * torch.sum((1 - gt), dim=(1,2))

    # Compute modified Dice coefficient
    modified_dice = 2 * torch.sum(w * w_intersection) / (torch.sum(w * w_union) + smooth)

    # Compute Dice loss
    loss = 1 - modified_dice

    return loss


def DiceLoss(inputs, targets, reduction='sum'):
#     inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    targets = targets.flatten(1)

    assert inputs.shape == targets.shape
    numerator = 2 * (inputs * targets).sum(1)
    denominator = (inputs * inputs).sum(-1) + (targets * targets).sum(-1)
    loss = 1 - (numerator) / (denominator + 1e-4)
    if reduction == 'none':
        return loss
    return loss.sum()



def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


sigmoid_focal_loss_jit: "torch.jit.ScriptModule" = torch.jit.script(sigmoid_focal_loss)


def sigmoid_focal_loss_hdetr(
    inputs, targets, num_masks, alpha: float = 0.25, gamma: float = 2
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_masks




def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    hw = inputs.shape[1]

    pos = F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    )
    neg = F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    )

    loss = torch.einsum("nc,mc->nm", pos, targets)
    loss = loss + torch.einsum("nc,mc->nm", neg, (1 - targets))

    return loss / hw


batch_sigmoid_ce_loss_jit = torch.jit.script(
    batch_sigmoid_ce_loss
)  # type: torch.jit.ScriptModule




def prob_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.1,
    gamma: float = 2,
    reduction: str = "none",
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    p = inputs
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


prob_focal_loss_jit: "torch.jit.ScriptModule" = torch.jit.script(prob_focal_loss)



def instance_kernel_loss(kernel):
    BN, D = kernel.size()
    
    if torch.all(torch.eq(kernel, 0)):
        return torch.tensor(1.0)
    
    kernel_normalized = F.normalize(kernel, dim=1)
    similarity = torch.mm(kernel_normalized, kernel_normalized.t())
    
    mask = torch.eye(BN, device=kernel.device)
    similarity_masked = similarity * (1 - mask)
    
    similarity_masked = torch.abs(similarity_masked)
    
    num_pairs = (BN) * (BN - 1) 
    mean_similarity = torch.sum(similarity_masked) / num_pairs
    loss = mean_similarity
    
    return loss


def cosine_similarity_kernel_loss(kernel_1, kernel_2):
    BN, D = kernel_1.size()
    
    if torch.all(torch.eq(kernel_1, 0)) and torch.all(torch.eq(kernel_2, 0)):
        return torch.tensor(1.0)
    
    kernel_1_normalized = F.normalize(kernel_1, dim=1)
    kernel_2_normalized = F.normalize(kernel_2, dim=1)
    similarity = torch.mm(kernel_1_normalized, kernel_2_normalized.t())
    
    similarity = torch.abs(similarity)
    
    mean_similarity = torch.mean(similarity)
    loss = mean_similarity
    
    return loss