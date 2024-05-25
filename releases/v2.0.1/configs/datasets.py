from os.path import join
from .base import Project, Image, dict
from utils.registry import Registry


class COCODataset:
    name: str
    coco_dataset: str


class Brightfield(COCODataset):
    name: str = 'brightfield'
    coco_dataset: str      = join(Project.work_dir, f'coco/{Project.project_id}', 'result.json')

    # ---------------
    # original data
    bf_images: str         = join(Project.work_dir, f'np/plane_images/{Project.project_id}/bf_lower_higher')
    bf_image_name: str     = f'X_{Image.size}_bf_lower_higher_v1'
    masks: str             = join(Project.work_dir, f'np/multi_headed_segmentation/4_channel_segmentation/{Project.project_id}')
    masks_name: str        = f'Y_{Image.size}_bordered_masks_bsz-4_v1'

    # ---------------
    # flow maps
    flow_masks: str        = join(Project.work_dir, f'np/overlap_segmentation/flow_maps')
    flow_masks_name: str   = f'Y_{Image.size}_flow_grad_map_[cellpose]'

    # ---------------
    # fl
    fl_masks: str          = '/gpfs/space/projects/PerkinElmer/exhaustive_dataset/exhaustive_dataset_gt/acapella/63x_water_nc_41FOV_NIH3T3/Nuclei'
    dataset_x63_dir: str   = join('/gpfs/space/projects/PerkinElmer/exhaustive_dataset/PhaseImagesDL/', 'e696ed04-bec0-4061-adc4-4ee935973439/')
    csv_dataset_dir: str   = join(dataset_x63_dir, '63x_water_nc_41FOV.csv')


class Brightfield_Nuc(Brightfield):
    name: str = 'brightfield_nuc'


class Synthetic_Brightfield(Brightfield):
    name: str = 'synthetic_brightfield'
    
    images: str = "/gpfs/space/home/prytula/data/datasets/cytoplasm_segmentation/synthetic_brightfield/[1024x1024]_[bf]_[not_normalized]_[aug4_scale]_[29.05.23]/images"
    masks: str  = "/gpfs/space/home/prytula/data/datasets/cytoplasm_segmentation/synthetic_brightfield/[1024x1024]_[bf]_[not_normalized]_[aug4_scale]_[29.05.23]/masks"



class OriginalPlusSynthetic_Brightfield(Brightfield):
    name: str = 'original_plus_synthetic_brightfield'
    
    coco_dataset: str = join(Project.work_dir, f'coco/{Project.project_id}', 'result.json')

    images: str = "/gpfs/space/home/prytula/data/datasets/cytoplasm_segmentation/synthetic_brightfield/[1024x1024]_[bf]_[not_normalized]_[aug4_scale]_[29.05.23]/images"
    masks: str  = "/gpfs/space/home/prytula/data/datasets/cytoplasm_segmentation/synthetic_brightfield/[1024x1024]_[bf]_[not_normalized]_[aug4_scale]_[29.05.23]/masks"


# ==================== EVICAN ====================
class EVICAN2(COCODataset):
    name: str = 'EVICAN2'
    type: str = 'EVICAN2'
    
    data_root = join("/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/", name)

    # valid_dataset=dict(
    #     images=join(data_root, "images/EVICAN_train2019"),
    #     ann_file=join(data_root, "annotations/EVICAN2/instances_train2019_EVICAN2.json")
    # )

    # valid_dataset=dict(
    #     images=join(data_root, "images/EVICAN_val2019"),
    #     ann_file=join(data_root, "annotations/EVICAN2/instances_val2019_EVICAN2.json")
    # )
    train_dataset=dict(
        images=join(data_root, "images/EVICAN_eval2019"),
        ann_file=join(data_root, "annotations/EVICAN2/instances_eval2019_easy_EVICAN2.json")
    )
    valid_dataset=dict(
        images=join(data_root, "images/EVICAN_eval2019"),
        ann_file=join(data_root, "annotations/EVICAN2/instances_eval2019_difficult_EVICAN2.json")
    )



class EVICAN2Easy(EVICAN2):
    name: str = 'EVICAN2Easy'

    eval_dataset=dict(
        images=join(EVICAN2.data_root, "images/EVICAN_eval2019"),
        ann_file=join(EVICAN2.data_root, "annotations/EVICAN2/instances_eval2019_easy_EVICAN2.json")
    )


class EVICAN2Medium(EVICAN2):
    name: str = 'EVICAN2Medium'
    eval_dataset=dict(
        images=join(EVICAN2.data_root, "images/EVICAN_eval2019"),
        ann_file=join(EVICAN2.data_root, "annotations/EVICAN2/instances_eval2019_medium_EVICAN2.json")
    )


class EVICAN2Difficult(EVICAN2):
    name: str = 'EVICAN2Difficult'
    eval_dataset=dict(
        images=join(EVICAN2.data_root, "images/EVICAN_eval2019"),
        ann_file=join(EVICAN2.data_root, "annotations/EVICAN2/instances_eval2019_difficult_EVICAN2.json")
    )



# ==================== LIVECELL ====================
class LiveCell(COCODataset):
    name: str = 'LiveCell'
    
    data_root = join("/gpfs/space/home/prytula/data/datasets/", name)

    train_dataset=dict(
        images=join(data_root, "images/livecell_train_val_images"),
        ann_file=join(data_root, "annotations/livecell_coco_train.json")
    )

    valid_dataset=dict(
        images=join(data_root, "images/livecell_train_val_images"),
        ann_file=join(data_root, "annotations/livecell_coco_val.json")
    )

    eval_dataset=dict(
        images=join(data_root, "images/livecell_test_images"),
        ann_file=join(data_root, "annotations/livecell_coco_test.json")
    )


class LiveCell2Percent(LiveCell):
    name: str = 'LiveCell2Percent'

    train_dataset=dict(
        images=join(LiveCell.data_root, "images/livecell_train_val_images"),
        ann_file=join(LiveCell.data_root, "annotations/0_train2percent.json")
    )

class LiveCell30Images(LiveCell2Percent):
    name: str = 'LiveCell30Images'






class Rectangle(COCODataset):
    name: str = 'rectangle'
    coco_dataset: str      = join(Project.home_dir, f'data/datasets/synthetic_datasets/rectangle/rectangles_[n=100]_[R_min=5_R_max=15]_[30.06.23].json')
    # coco_dataset: str      = join(Project.home_dir, f'data/datasets/synthetic_datasets/rectangle/rectangles_[S_min=20_S_max=300]_[n=1000]_[R_min=2_R_max=15]_[overlap=0.5]_[15.08.23].json')
    # coco_dataset: str      = join(Project.home_dir, f'data/datasets/synthetic_datasets/rectangle/rectangles_[S_min=50_S_max=200]_[n=1000]_[R_min=2_R_max=3]_[overlap=0.3-0.8]_[12.09.23].json')




DATASETS_CFG = Registry("datasets_cfg")

# ==================== Exhaustive ====================
DATASETS_CFG.register(Brightfield.name, Brightfield)
DATASETS_CFG.register(OriginalPlusSynthetic_Brightfield.name, OriginalPlusSynthetic_Brightfield)
DATASETS_CFG.register(Rectangle.name, Rectangle)


# ==================== EVICAN ====================
DATASETS_CFG.register(EVICAN2.name, EVICAN2)
DATASETS_CFG.register(EVICAN2Easy.name, EVICAN2Easy)
DATASETS_CFG.register(EVICAN2Medium.name, EVICAN2Medium)
DATASETS_CFG.register(EVICAN2Difficult.name, EVICAN2Difficult)


# ==================== LIVECELL ====================
DATASETS_CFG.register(LiveCell.name, LiveCell)
DATASETS_CFG.register(LiveCell2Percent.name, LiveCell2Percent)
DATASETS_CFG.register(LiveCell30Images.name, LiveCell30Images)