import os
from os.path import join
from datetime import datetime
import inspect
import json 
import yaml


TIME = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
JOB_ID = os.environ.get('SLURM_JOB_ID')


class BaseConfig:
    @staticmethod
    def get_class_attributes(class_obj):
        attributes = {}
        class_members = inspect.getmembers(class_obj)
        for member_name, member_value in class_members:
            if not member_name.startswith('__') and not inspect.ismethod(member_value):
                attributes[member_name] = member_value
        return attributes

    @staticmethod
    def create_cfg_dict():
        cfg_dict = {}
        cfg_members = inspect.getmembers(cfg)
        for member_name, member_value in cfg_members:
            if inspect.isclass(member_value):
                attributes = cfg.get_class_attributes(member_value)
                cfg_dict[member_name] = attributes
        return cfg_dict

    def __dict__(self):
        cfg_dict = cfg.create_cfg_dict()
        cfg_dict = {k: v for k, v in cfg_dict.items() if not k.startswith('__')}
        return cfg_dict
    
    def __repr__(self):
        return str(json.dumps(self.__dict__(), sort_keys=True, indent=4))
        # return str(self.__dict__())

        # attrs = {}
        # for attr, value in self.__dict__.items():
        #     if isinstance(value, type):
        #         attrs[attr] = repr(value())
        #     else:
        #         attrs[attr] = value
        # return str(attrs)
    
    @classmethod
    def yaml_load(cls, file_path):
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)

        cls._update_attributes(cls, data)

    @staticmethod
    def _update_attributes(obj, data):
        for key, value in data.items():
            if hasattr(obj, key):
                if isinstance(value, dict):
                    nested_obj = getattr(obj, key)
                    cfg._update_attributes(nested_obj, value)
                else:
                    setattr(obj, key, value)



class Image:
    size: int   = 1080

# Linking datasets
class Project:
    home_dir: str    = '/gpfs/space/home/prytula'
    work_dir: str    = join(home_dir, 'data/mask_labels/x63_fl') # home_dir + data_dir
    project_id: str = 'project-2-at-2022-05-20-09-23-83cd15f1'
        
            
class Train:
    epochs: int     = 1001
    n_folds: int    = 5
    size: int       = [512, 512]
    bs: int         = 2
    lr: float       = 1e-4
    # wd: float       = 1e-5
    wd: float       = 0.05
    augment: bool   = True
    

class Valid:
    size: int       = [512, 512]
    bs: int         = 2
        
        
class Model:
    arch: str         = 'sparse_seunet'
    in_channels: int  = 1
    out_channels: int = 1

    num_groups: int   = 1
    num_classes: int  = 1

    num_convs: int    = 4
    n_levels: int     = 5

    # instance head.
    kernel_dim: int   = 128
    num_masks: int    = 25

    # model structure.
    coord_conv: bool  = True
    multi_level: bool = True
    
    # weights.
    load_pretrained: bool = True
    weights: str          = "/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=45867499]-[2023-08-07 11:44:24]/checkpoints/best.pth"
    save_checkpoint: bool = False
    save_model_files: bool = True

    # TODO: load config from experiment + param overloading
    # experimental.
    load_from_files: bool = False
    model_files: str      = "/gpfs/space/home/prytula/scripts/experimental_segmentation/SparseUnet/runs/[sparse_seunet]-[512]/[brightfield]/[softmax_iam]/[multi_level=True]-[coord_conv=True]-[losses=['labels', 'masks']]/[job=45867499]-[2023-08-07 11:44:24]/model_files"

    # losses.
    losses: list = ["labels", "masks"]
    # losses: list = ["labels", "masks", "iam"]
    # losses: list = ["labels", "masks", "overlaps"]



class Run:
    runs_dir: str           = 'runs'
    experiment_name: str    = f'[{Model.arch}]-[512]'
    run_name: str           = f'[<dataset_name>]/[softmax_iam]/[multi_level={Model.multi_level}]-[coord_conv={Model.coord_conv}]-[losses={Model.losses}]/[job={JOB_ID}]-[{TIME}]'
    exist_ok: bool          = False
    comment: str            = """
                                - group iams (parallel iam convs for instance and overlap features)
                                - overlaps
                                - added BatchNorm2d to stacked 3x3 convolutions 
                                - fixed sigmoid iam 
                                - new wd 
                                - removed iam matcher, now we match preds with gt and use indices from matched preds to guide correcponding iams
                              """
        

class cfg(BaseConfig):
    model: Model           = Model
    train: Train           = Train
    valid: Valid           = Valid
    project: Project       = Project
    image: Image           = Image
    run: Run               = Run
    # dataset: str           = "brightfield"
    dataset: str           = "<dataset_name>"
    
    seed: int = 3407
    device = "cuda"
    
    scheduler = 'CosineAnnealingLR'
    min_lr = 1e-6
    T_max = 55
    T_0 = 25
    warmup_epochs = 10

    verbose: bool = True
    

cfg = cfg()
# print(cfg)

# from utils import Config
# cfg = Config(cfg().__dict__())
# print(cfg)

# print(cfg.__dict__)
# print(cfg().__dict__())

