import torch
import sys
sys.path.append("./")
from models.seg.loss import compute_mask_iou
from utils.registry import CRITERIONS
from configs import cfg


def create_rectangular_mask(dimensions, rectangle):
    """Create a mask of given dimensions with a rectangular area set to 1."""
    mask = torch.zeros(dimensions)
    mask[rectangle[1]:rectangle[3], rectangle[0]:rectangle[2]] = 1
    return mask

def mock_data():
    outputs = {
        "pred_logits": torch.randn(1, 5, 2),
        "pred_scores": torch.tensor([[0.9, 0.8, 0.1, 0.2, 0.85]]),
        "pred_masks": torch.stack([
            create_rectangular_mask((10, 10), (2, 2, 5, 5)),
            create_rectangular_mask((10, 10), (5, 5, 8, 8)),
            create_rectangular_mask((10, 10), (0, 0, 3, 3)),
            create_rectangular_mask((10, 10), (7, 7, 9, 9)),
            create_rectangular_mask((10, 10), (1, 1, 4, 4))
        ]).unsqueeze(0),  # Add batch dimension
        "pred_bboxes": torch.tensor([[[3.5, 3.5, 3, 3], [6.5, 6.5, 3, 3], [1.5, 1.5, 3, 3], [8, 8, 2, 2], [2.5, 2.5, 3, 3]]])
    }

    targets = [{
        "masks": torch.stack([
            create_rectangular_mask((10, 10), (1, 1, 4, 4)),
            create_rectangular_mask((10, 10), (3, 3, 6, 6)),
            create_rectangular_mask((10, 10), (5, 5, 8, 8)),
        ]),
        "labels": torch.randint(0, 1, (3,)),
        "bboxes": torch.tensor([[2.5, 2.5, 3, 3], [4.5, 4.5, 3, 3], [6.5, 6.5, 3, 3]])
    }]

    indices = [(torch.tensor([0, 2, 4]), torch.tensor([0, 1, 2]))]

    return outputs, targets, indices


def test():
    criterion = CRITERIONS.build(cfg.model.criterion)
    outputs, targets, indices = mock_data()

    indices = criterion.matcher(outputs, targets, (10, 10))
    print(indices)

    num_masks = sum(len(t['masks']) for t in targets)  # Calculate total number of masks across all targets
    loss_results = criterion.loss_labels(outputs, targets, indices, num_masks, (10, 10))
    print(loss_results)


test()
