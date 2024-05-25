**TODO**: 
  - datasets benchmark 
    - sequentially evaluate the model over specified datasets

  - add dataset cfg to Dataset


**IDEA**: 
  - given the instance kernel, we can regress the object boundary and the object related overlap
  - 🔥 overlap weight map for instances

  - evaluate on occluded dataset objects
    - filter gt -> match pred => if there is an improvement there exists best segmentation


  - using sparse-unet to classify samples - we can use iam maps maps for as attention maps
    - cls comes from iams


🚀 **EXPERIMENTS**:
  - 🔥 indentify the problem with bad training
    - retrain on sparse_seunet_occluder
    - retrain on sparse_seunet_occluder_gcn_mh
      - remove feature addition
      - revert prior_branches back to original state
    

  - 🔁 prepare synthetic dataset
  - 🔁 prepare prob maps


