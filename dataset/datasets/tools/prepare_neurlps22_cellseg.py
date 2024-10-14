import os
from os.path import join
import cv2
import json
import random
import numpy as np

from tqdm import tqdm
from datetime import datetime
from skimage import measure
import pycocotools.mask as coco_mask


def filter_empty_masks(masks):
    kept_indices = np.any(masks, axis=(0, 1))
    masks = masks[..., kept_indices]
    return masks, kept_indices

def get_masks(mask):
    unique_values = np.unique(mask)
    unique_values = unique_values[unique_values != 0]
    
    instance_masks = []
    for value in unique_values:
        instance_mask = (mask == value).astype(np.uint8)
        instance_masks.append(instance_mask)
    
    if len(instance_masks) > 0:
        instance_masks = np.stack(instance_masks, axis=-1)
    else:
        instance_masks = np.zeros((mask.shape[0], mask.shape[1], 0))

    return instance_masks


def close_contour(contour):
    if not np.array_equal(contour[0], contour[-1]):
        contour = np.vstack((contour, contour[0]))
    return contour

def binary_mask_to_polygon(binary_mask, tolerance=1):
    polygons = []
    padded_binary_mask = np.pad(binary_mask, pad_width=1, mode='constant', constant_values=0)
    contours = measure.find_contours(padded_binary_mask, 0.5)
    
    for contour in contours:
        if contour.shape[0] < 3:  # Filter out too small contours
            continue
        contour = contour - 1  # Adjust contour coordinates to original image size
        contour = close_contour(contour)
        contour = measure.approximate_polygon(contour, tolerance)
        if len(contour) < 3:  # Valid polygons with at least 3 points
            continue
        contour = np.flip(contour, axis=1)
        segmentation = contour.ravel().tolist()
        segmentation = [max(0, i) for i in segmentation]  # Ensure no negative values
        polygons.append(segmentation)
    
    return polygons

def create_image_info(image_id, width, height, file_name):
    return {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
        "date_captured": datetime.utcnow().isoformat(' '),
        "license": 1,
        "coco_url": "",
        "flickr_url": ""
    }

def create_annotation_info(annotation_id, image_id, category_id, binary_mask):
    binary_mask_encoded = coco_mask.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    area = coco_mask.area(binary_mask_encoded)
    bbox = coco_mask.toBbox(binary_mask_encoded).tolist()
    segmentation = binary_mask_to_polygon(binary_mask)
    if not segmentation:
        return None
    return {
        "id": annotation_id,
        "image_id": image_id,
        "category_id": category_id,
        "iscrowd": 0,
        "area": int(area),
        "bbox": bbox,
        "segmentation": segmentation,
        "width": binary_mask.shape[1],
        "height": binary_mask.shape[0],
    }

def masks2coco(masks, image_id, category_id, ann_id_last, file_name):
    annotations = []
    image_info = create_image_info(image_id, masks.shape[1], masks.shape[0], file_name)
    for i in range(masks.shape[-1]):
        mask = masks[..., i]
        annotation_info = create_annotation_info(ann_id_last, image_id, category_id, mask)
        if annotation_info:
            annotations.append(annotation_info)
            ann_id_last += 1
    return {"images": [image_info], "annotations": annotations}, ann_id_last

def save_image(image, save_path, file_name, extension):
    os.makedirs(save_path, exist_ok=True)
    img_png_path = os.path.join(save_path, os.path.splitext(file_name)[0] + ".png")
    cv2.imwrite(img_png_path, image)
    img_name = os.path.splitext(file_name)[0] + ".png"
    return img_name


def prepare_coco(image_ids, labels_dir, images_dir, save_path, category_id, ann_id_last, split_name):
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {
                "supercategory": "cell",
                "id": 1,
                "name": "cell"
            }
        ]
    }
    os.makedirs(f"{save_path}/images", exist_ok=True)
    os.makedirs(f"{save_path}/annotations", exist_ok=True)
    
    skips = 0
    for idx in tqdm(image_ids, desc=f"Processing {split_name}"):
        file_name = None
        extensions = ['png', 'jpg', 'tif', 'tiff', 'bmp']
        for ext in extensions:
            img_path = join(images_dir, f'cell_{idx:05d}.{ext}')
            if os.path.exists(img_path):
                file_name = f'cell_{idx:05d}.{ext}'
                break
        if file_name is None:
            print(f"Image for ID {idx} not found!")
            continue

        label_path = join(labels_dir, f'cell_{idx:05d}_label.tiff')

        img = cv2.imread(img_path, -1)
        mask = cv2.imread(label_path, -1)

        img = img.astype(np.float32)
        img /= img.max()
        img *= 255.

        if mask.max() > 100:
            skips += 1
            continue

        masks = get_masks(mask)
        masks, _ = filter_empty_masks(masks)
        img_name = save_image(img, f"{save_path}/images", file_name, extension="png")

        coco_image, ann_id_last = masks2coco(masks, idx, category_id, ann_id_last, img_name)
        coco_data["images"].extend(coco_image["images"])
        coco_data["annotations"].extend(coco_image["annotations"])

    with open(f'{save_path}/annotations/{split_name}.json', 'w') as f:
        json.dump(coco_data, f, indent=4)

    print(f'skipped {skips} images.')
    print(f'Saved COCO {split_name} annotations to {save_path}/{split_name}.json')


def prepare_neurlps22_cellseg(images_dir, labels_dir, save_path, split_ratios=(60, 10, 30)):
    all_image_ids = list(range(1, 1001))
    random.shuffle(all_image_ids)

    n_train = int(split_ratios[0] / 100 * len(all_image_ids))
    n_valid = int(split_ratios[1] / 100 * len(all_image_ids))
    n_test = len(all_image_ids) - n_train - n_valid

    # train_ids = all_image_ids[:n_train]
    # valid_ids = all_image_ids[n_train:n_train + n_valid]
    # test_ids = all_image_ids[n_train + n_valid:]

    train_ids = [949, 200, 981, 481, 210, 622, 96, 835, 502, 182, 455, 474, 501, 537, 869, 277, 68, 556, 231, 486, 580, 178, 484, 585, 909, 395, 272, 724, 603, 427, 582, 685, 573, 772, 918, 865, 506, 186, 460, 908, 529, 380, 920, 135, 830, 681, 241, 93, 767, 523, 756, 919, 933, 487, 147, 215, 723, 936, 584, 153, 270, 945, 823, 522, 345, 965, 26, 706, 316, 749, 816, 329, 58, 45, 825, 283, 17, 76, 403, 480, 971, 549, 462, 853, 984, 16, 600, 831, 333, 330, 146, 271, 847, 898, 641, 332, 252, 913, 596, 504, 925, 352, 232, 356, 885, 435, 464, 148, 61, 720, 442, 699, 625, 910, 896, 312, 786, 750, 588, 306, 728, 645, 304, 482, 154, 889, 323, 179, 942, 82, 764, 336, 70, 214, 680, 861, 822, 593, 552, 583, 201, 388, 376, 833, 950, 424, 340, 54, 632, 286, 361, 904, 862, 430, 483, 438, 779, 698, 473, 402, 990, 855, 517, 881, 63, 788, 28, 373, 111, 906, 672, 109, 307, 247, 32, 528, 443, 664, 554, 263, 951, 614, 659, 353, 511, 635, 188, 498, 988, 661, 616, 623, 757, 40, 863, 3, 866, 29, 288, 891, 208, 708, 320, 117, 515, 895, 396, 785, 707, 611, 289, 325, 172, 655, 524, 95, 838, 547, 527, 938, 104, 960, 38, 2, 185, 741, 7, 673, 75, 94, 969, 682, 565, 982, 240, 804, 397, 540, 12, 257, 619, 98, 534, 365, 381, 116, 268, 647, 631, 207, 689, 952, 297, 158, 705, 266, 624, 897, 979, 533, 407, 784, 697, 243, 359, 662, 799, 245, 587, 221, 366, 618, 903, 181, 256, 754, 296, 191, 832, 401, 477, 351, 391, 693, 450, 795, 829, 492, 770, 398, 828, 797, 649, 735, 930, 923, 808, 422, 220, 157, 513, 230, 159, 226, 479, 679, 190, 551, 139, 10, 557, 355, 301, 31, 8, 281, 97, 514, 560, 35, 370, 636, 916, 433, 574, 837, 817, 644, 293, 273, 295, 434, 710, 497, 494, 508, 144, 156, 73, 223, 56, 280, 390, 468, 987, 646, 213, 130, 500, 328, 873, 132, 374, 101, 42, 986, 163, 712, 536, 567, 128, 342, 110, 99, 998, 628, 219, 883, 842, 375, 851, 733, 209, 884, 189, 718, 691, 774, 843, 571, 507, 490, 864, 633, 926, 963, 846, 563, 946, 512, 957, 937, 74, 134, 463, 943, 546, 90, 856, 278, 227, 260, 993, 453, 100, 545, 276, 428, 651, 165, 674, 150, 198, 815, 458, 970, 437, 572, 734, 175, 850, 598, 739, 944, 66, 39, 421, 539, 385, 496, 222, 819, 357, 18, 703, 197, 929, 530, 690, 839, 686, 124, 996, 870, 912, 907, 989, 606, 15, 905, 940, 322, 339, 107, 911, 363, 966, 488, 33, 821, 86, 212, 308, 924, 751, 931, 184, 813, 521, 592, 576, 738, 416, 151, 218, 43, 711, 525, 334, 648, 489, 962, 369, 510, 71, 755, 845, 964, 118, 318, 736, 704, 30, 777, 205, 722, 629, 457, 719, 255, 744, 64, 678, 595, 701, 887, 217, 650, 878, 138, 569, 726, 180, 983, 836, 761, 642, 605, 393, 818, 472, 882, 802, 387, 531, 173, 612, 848, 88, 602, 542, 80, 499, 417, 671, 21, 466, 771, 590, 867, 145, 409, 19, 199, 14, 389, 597, 444, 404, 384, 995, 694, 491, 778, 715, 274, 753, 658, 977, 503, 746, 810, 383, 928, 164, 947, 568, 742, 49, 792, 5, 991, 890, 358, 341, 265, 267, 196, 349, 203, 337, 740, 160, 386, 999, 872, 758, 591, 800, 879, 894, 953, 174, 550, 858, 211, 290, 343, 67, 228]
    valid_ids = [445, 346, 752, 627, 406, 91, 246, 121, 613, 25, 809, 108, 967, 224, 532, 299, 805, 730, 354, 859, 102, 760, 766, 264, 244, 558, 994, 608, 781, 509, 170, 162, 461, 237, 269, 660, 857, 824, 1000, 934, 287, 79, 394, 776, 676, 958, 291, 675, 683, 604, 731, 127, 801, 841, 927, 643, 187, 939, 258, 716, 634, 790, 902, 663, 493, 11, 368, 302, 364, 997, 314, 254, 630, 976, 294, 566, 348, 955, 262, 888, 854, 796, 279, 763, 978, 840, 561, 305, 469, 52, 311, 168, 581, 379, 350, 140, 657, 544, 125, 454]
    test_ids = [87, 310, 194, 119, 688, 136, 34, 917, 806, 575, 131, 48, 570, 451, 700, 875, 27, 202, 654, 77, 161, 47, 972, 282, 53, 871, 452, 747, 382, 115, 55, 259, 668, 505, 92, 948, 959, 169, 667, 321, 607, 692, 261, 470, 6, 725, 475, 447, 65, 844, 732, 768, 702, 123, 195, 961, 932, 367, 586, 233, 985, 954, 789, 814, 639, 578, 408, 309, 177, 610, 229, 471, 941, 236, 36, 126, 44, 206, 400, 360, 377, 782, 78, 849, 440, 85, 412, 968, 729, 553, 176, 476, 225, 251, 617, 432, 852, 204, 677, 787, 562, 620, 807, 216, 518, 9, 899, 478, 239, 415, 133, 112, 59, 526, 242, 338, 860, 448, 886, 765, 743, 171, 84, 167, 50, 653, 141, 89, 935, 548, 465, 193, 142, 717, 656, 13, 727, 23, 57, 467, 129, 874, 519, 192, 249, 974, 418, 69, 834, 22, 60, 152, 371, 684, 783, 880, 495, 666, 331, 405, 621, 41, 811, 326, 812, 446, 615, 419, 579, 820, 687, 248, 900, 275, 420, 759, 914, 238, 535, 137, 520, 922, 665, 876, 980, 362, 183, 973, 426, 775, 794, 737, 122, 745, 868, 543, 439, 992, 324, 640, 1, 638, 313, 798, 601, 344, 51, 538, 791, 410, 166, 347, 436, 114, 594, 4, 24, 901, 826, 599, 714, 423, 748, 516, 459, 456, 793, 303, 335, 315, 626, 780, 589, 106, 319, 559, 298, 414, 392, 317, 250, 378, 253, 113, 609, 103, 915, 37, 441, 20, 577, 120, 975, 637, 541, 399, 769, 652, 762, 411, 413, 155, 773, 892, 372, 235, 285, 234, 81, 149, 713, 83, 893, 431, 62, 956, 72, 695, 284, 327, 425, 921, 803, 292, 696, 721, 46, 877, 564, 143, 485, 105, 300, 709, 429, 669, 555, 449, 827, 670]


    print(f"Train set: \n{train_ids}\n")
    print(f"Valid set: \n{valid_ids}\n")
    print(f"Test set: \n{test_ids}\n")

    prepare_coco(train_ids, labels_dir, images_dir, save_path, 
                 category_id=1, ann_id_last=0, split_name="train")
    prepare_coco(valid_ids, labels_dir, images_dir, save_path, 
                 category_id=1, ann_id_last=len(train_ids), split_name="valid")
    prepare_coco(test_ids, labels_dir, images_dir, save_path, 
                 category_id=1, ann_id_last=len(train_ids) + len(valid_ids), split_name="test")


if __name__ == "__main__":
    data_root = '/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/NeurlPS22-CellSeg'
    images_dir = join(data_root, "images")
    labels_dir = join(data_root, "labels")
    save_path = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/NeurlPS22-CellSeg/coco_new1"

    prepare_neurlps22_cellseg(images_dir, labels_dir, save_path)



# Train set: 
# [949, 200, 981, 481, 210, 622, 96, 835, 502, 182, 455, 474, 501, 537, 869, 277, 68, 556, 231, 486, 580, 178, 484, 585, 909, 395, 272, 724, 603, 427, 582, 685, 573, 772, 918, 865, 506, 186, 460, 908, 529, 380, 920, 135, 830, 681, 241, 93, 767, 523, 756, 919, 933, 487, 147, 215, 723, 936, 584, 153, 270, 945, 823, 522, 345, 965, 26, 706, 316, 749, 816, 329, 58, 45, 825, 283, 17, 76, 403, 480, 971, 549, 462, 853, 984, 16, 600, 831, 333, 330, 146, 271, 847, 898, 641, 332, 252, 913, 596, 504, 925, 352, 232, 356, 885, 435, 464, 148, 61, 720, 442, 699, 625, 910, 896, 312, 786, 750, 588, 306, 728, 645, 304, 482, 154, 889, 323, 179, 942, 82, 764, 336, 70, 214, 680, 861, 822, 593, 552, 583, 201, 388, 376, 833, 950, 424, 340, 54, 632, 286, 361, 904, 862, 430, 483, 438, 779, 698, 473, 402, 990, 855, 517, 881, 63, 788, 28, 373, 111, 906, 672, 109, 307, 247, 32, 528, 443, 664, 554, 263, 951, 614, 659, 353, 511, 635, 188, 498, 988, 661, 616, 623, 757, 40, 863, 3, 866, 29, 288, 891, 208, 708, 320, 117, 515, 895, 396, 785, 707, 611, 289, 325, 172, 655, 524, 95, 838, 547, 527, 938, 104, 960, 38, 2, 185, 741, 7, 673, 75, 94, 969, 682, 565, 982, 240, 804, 397, 540, 12, 257, 619, 98, 534, 365, 381, 116, 268, 647, 631, 207, 689, 952, 297, 158, 705, 266, 624, 897, 979, 533, 407, 784, 697, 243, 359, 662, 799, 245, 587, 221, 366, 618, 903, 181, 256, 754, 296, 191, 832, 401, 477, 351, 391, 693, 450, 795, 829, 492, 770, 398, 828, 797, 649, 735, 930, 923, 808, 422, 220, 157, 513, 230, 159, 226, 479, 679, 190, 551, 139, 10, 557, 355, 301, 31, 8, 281, 97, 514, 560, 35, 370, 636, 916, 433, 574, 837, 817, 644, 293, 273, 295, 434, 710, 497, 494, 508, 144, 156, 73, 223, 56, 280, 390, 468, 987, 646, 213, 130, 500, 328, 873, 132, 374, 101, 42, 986, 163, 712, 536, 567, 128, 342, 110, 99, 998, 628, 219, 883, 842, 375, 851, 733, 209, 884, 189, 718, 691, 774, 843, 571, 507, 490, 864, 633, 926, 963, 846, 563, 946, 512, 957, 937, 74, 134, 463, 943, 546, 90, 856, 278, 227, 260, 993, 453, 100, 545, 276, 428, 651, 165, 674, 150, 198, 815, 458, 970, 437, 572, 734, 175, 850, 598, 739, 944, 66, 39, 421, 539, 385, 496, 222, 819, 357, 18, 703, 197, 929, 530, 690, 839, 686, 124, 996, 870, 912, 907, 989, 606, 15, 905, 940, 322, 339, 107, 911, 363, 966, 488, 33, 821, 86, 212, 308, 924, 751, 931, 184, 813, 521, 592, 576, 738, 416, 151, 218, 43, 711, 525, 334, 648, 489, 962, 369, 510, 71, 755, 845, 964, 118, 318, 736, 704, 30, 777, 205, 722, 629, 457, 719, 255, 744, 64, 678, 595, 701, 887, 217, 650, 878, 138, 569, 726, 180, 983, 836, 761, 642, 605, 393, 818, 472, 882, 802, 387, 531, 173, 612, 848, 88, 602, 542, 80, 499, 417, 671, 21, 466, 771, 590, 867, 145, 409, 19, 199, 14, 389, 597, 444, 404, 384, 995, 694, 491, 778, 715, 274, 753, 658, 977, 503, 746, 810, 383, 928, 164, 947, 568, 742, 49, 792, 5, 991, 890, 358, 341, 265, 267, 196, 349, 203, 337, 740, 160, 386, 999, 872, 758, 591, 800, 879, 894, 953, 174, 550, 858, 211, 290, 343, 67, 228]

# Valid set: 
# [445, 346, 752, 627, 406, 91, 246, 121, 613, 25, 809, 108, 967, 224, 532, 299, 805, 730, 354, 859, 102, 760, 766, 264, 244, 558, 994, 608, 781, 509, 170, 162, 461, 237, 269, 660, 857, 824, 1000, 934, 287, 79, 394, 776, 676, 958, 291, 675, 683, 604, 731, 127, 801, 841, 927, 643, 187, 939, 258, 716, 634, 790, 902, 663, 493, 11, 368, 302, 364, 997, 314, 254, 630, 976, 294, 566, 348, 955, 262, 888, 854, 796, 279, 763, 978, 840, 561, 305, 469, 52, 311, 168, 581, 379, 350, 140, 657, 544, 125, 454]

# Test set: 
# [87, 310, 194, 119, 688, 136, 34, 917, 806, 575, 131, 48, 570, 451, 700, 875, 27, 202, 654, 77, 161, 47, 972, 282, 53, 871, 452, 747, 382, 115, 55, 259, 668, 505, 92, 948, 959, 169, 667, 321, 607, 692, 261, 470, 6, 725, 475, 447, 65, 844, 732, 768, 702, 123, 195, 961, 932, 367, 586, 233, 985, 954, 789, 814, 639, 578, 408, 309, 177, 610, 229, 471, 941, 236, 36, 126, 44, 206, 400, 360, 377, 782, 78, 849, 440, 85, 412, 968, 729, 553, 176, 476, 225, 251, 617, 432, 852, 204, 677, 787, 562, 620, 807, 216, 518, 9, 899, 478, 239, 415, 133, 112, 59, 526, 242, 338, 860, 448, 886, 765, 743, 171, 84, 167, 50, 653, 141, 89, 935, 548, 465, 193, 142, 717, 656, 13, 727, 23, 57, 467, 129, 874, 519, 192, 249, 974, 418, 69, 834, 22, 60, 152, 371, 684, 783, 880, 495, 666, 331, 405, 621, 41, 811, 326, 812, 446, 615, 419, 579, 820, 687, 248, 900, 275, 420, 759, 914, 238, 535, 137, 520, 922, 665, 876, 980, 362, 183, 973, 426, 775, 794, 737, 122, 745, 868, 543, 439, 992, 324, 640, 1, 638, 313, 798, 601, 344, 51, 538, 791, 410, 166, 347, 436, 114, 594, 4, 24, 901, 826, 599, 714, 423, 748, 516, 459, 456, 793, 303, 335, 315, 626, 780, 589, 106, 319, 559, 298, 414, 392, 317, 250, 378, 253, 113, 609, 103, 915, 37, 441, 20, 577, 120, 975, 637, 541, 399, 769, 652, 762, 411, 413, 155, 773, 892, 372, 235, 285, 234, 81, 149, 713, 83, 893, 431, 62, 956, 72, 695, 284, 327, 425, 921, 803, 292, 696, 721, 46, 877, 564, 143, 485, 105, 300, 709, 429, 669, 555, 449, 827, 670]




# import json
# import os

# data_root = '/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/NeurlPS22-CellSeg/coco/annotations/test.json'
# with open(data_root, 'r') as f:
#     coco_data = json.load(f)

# for image in coco_data['images']:
#     image['file_name'] = os.path.basename(image['file_name'])

# with open(data_root, 'w') as f:
#     json.dump(coco_data, f, indent=4)

# print("File names have been updated successfully.")
