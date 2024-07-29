from utils.visualise import visualize

def mask_nms(cate_labels, seg_masks, sum_masks, cate_scores, nms_thr=0.5):
    n_samples = len(cate_scores)
    if n_samples == 0:
        return []

    keep = seg_masks.new_ones(cate_scores.shape)
    seg_masks = seg_masks.float()

    for i in range(n_samples - 1):
        if not keep[i]:
            continue
        mask_i = seg_masks[i]
        label_i = cate_labels[i]

        for j in range(i + 1, n_samples, 1):
            if not keep[j]:
                continue
            mask_j = seg_masks[j]
            label_j = cate_labels[j]
            if label_i != label_j:
                continue

            # overlaps
            inter = (mask_i * mask_j).sum()
            union = sum_masks[i] + sum_masks[j] - inter
        
            if union > 0:
                if inter / union > nms_thr:
                    keep[j] = False
            else:
                keep[j] = False
    return keep