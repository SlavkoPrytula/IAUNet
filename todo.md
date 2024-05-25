- unify coco format datasets loading 
- fix coco visualization

- unify model inference class
    - abillity to have results = model.inference(data_samples) in the evaluator
    - basically this adds a unification to the evaluation of the model, ensuring same processing
    steps for all inference-like evaluations


-------
- unify the load_from_file imports
    - model class, losses and matchers should also load from pre-saved files

- _make_stack_3x3_convs
    - inner convs should operate on higher number of feature channels and do projection on exit
    - this means that we give the block an ability to make choices and see more features before final projection to N output kernels
