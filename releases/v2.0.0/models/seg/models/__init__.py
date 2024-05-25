from .sparse_seunet import SparseSEUnet as SparseSEUnet
from .sparse_seunet_add_overlaps import SparseSEUnet as SparseSEUnetAddOverlaps

from .sparse_seunet_occluder_v1 import SparseSEUnet as SparseSEUnetOccluder
from .sparse_seunet_occluder_overlap import SparseSEUnet as SparseSEUnetOccluderOverlap

from .sparse_seunet_occluder_gcn import SparseSEUnet as SparseSEUnetOccluderGCN
from .sparse_seunet_occluder_gcn_mh import SparseSEUnet as SparseSEUnetOccluderGCNMultiHeaded

from .sparse_seunet_occluder_ml import SparseSEUnet as SparseSEUnetOccluderMultiLevel

from .sparse_seunet_occluder_mh import SparseSEUnet as SparseSEUnetOccluderMultiHeaded

from .sparse_seunet_ml_iam import SparseSEUnet as SparseSEUnetMultiLevelIAM

from .sparse_seunet_simplified import SparseSEUnet as SparseSEUnetSimplified
# from .sparse_seunet_simplified_v1 import SparseSEUnet as SparseSEUnetSimplified
# from .hornet_uppernet import SparseSEUnet as SparseSEUnetSimplified

# from .hornet_uppernet import SparseSEUnet

from .custom.hornet.hornet_uppernet import SparseSEUnet
from .custom.hornet.hornet_uppernet_occluder import SparseSEUnet
from .custom.convnext.convnext_uppernet import SparseSEUnet


from .fixes.seunet.sparse_seunet import SparseSEUnet
from .fixes.seunet.sparse_seunet_larger import SparseSEUnet
# from .fixes.seunet.sparse_seunet_dwc import SparseSEUnet
from .sparse_seunet_dwc import SparseSEUnet
from .fixes.seunet.sparse_seunet_dwc_l import SparseSEUnet

from .sparse_seunet_gcn import SparseSEUnet as SparseSEUnet
from .sparse_seunet_decoupled import SparseSEUnet as SparseSEUnet

# def build_model(cfg):
