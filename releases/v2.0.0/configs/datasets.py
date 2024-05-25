from os.path import join
from .base import Project, Image


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




class Rectangle(COCODataset):
    name: str = 'rectangle'
    # coco_dataset: str      = join(Project.home_dir, f'data/datasets/synthetic_datasets/rectangle/rectangles_[n=100]_[R_min=5_R_max=15]_[30.06.23].json')
    # coco_dataset: str      = join(Project.home_dir, f'data/datasets/synthetic_datasets/rectangle/rectangles_[S_min=20_S_max=300]_[n=1000]_[R_min=2_R_max=15]_[overlap=0.5]_[15.08.23].json')
    coco_dataset: str      = join(Project.home_dir, f'data/datasets/synthetic_datasets/rectangle/rectangles_[S_min=50_S_max=200]_[n=1000]_[R_min=2_R_max=3]_[overlap=0.3-0.8]_[12.09.23].json')




from utils.registry import Registry

DATASETS_CFG = Registry("datasets_cfg")

DATASETS_CFG.register(Brightfield.name, Brightfield)
DATASETS_CFG.register(OriginalPlusSynthetic_Brightfield.name, OriginalPlusSynthetic_Brightfield)
DATASETS_CFG.register(Rectangle.name, Rectangle)