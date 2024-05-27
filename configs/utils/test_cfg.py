# import sys
# sys.path.append("./")

# from configs.base import cfg as base_cfg

# def merge_config_from_file(base_cfg, config_path):
#     base_cfg.merge_from_file(config_path)
#     # Update the references dynamically
#     base_cfg.update_from_dict(base_cfg.config_dict)

# # Load base config
# cfg = base_cfg

# # Load model-specific config and merge
# model_config_path = "configs/models/iaunet_resnet/iaunet_r50.py"
# merge_config_from_file(cfg, model_config_path)

# # Print the final config to verify
# cfg = cfg.__class__()
# print(cfg)



# class B:
#     name = "Original B"

# class A:
#     def __init__(self):
#         self.b: B = B()
#         self.c = B.name



# # -----
# def reinitialize_references(instance, new_class):
#     print()
#     for key, value in instance.__dict__.items():
#         print(type(value), new_class, type(value) == new_class)
#         if type(value) == new_class:
#             print(key, value)
    

#     instance.b = new_class() # how does it know its calss B?

# def merge_and_update(instance, new_class):
#     reinitialize_references(instance, new_class)

# # Base instance
# a = A()
# print("Before update:")
# print(f"a.b.name: {a.b.name}")
# print(f"a.c: {a.c}")

# # New class definition
# class B:
#     name = "Updated B"

# # Update and reinitialize
# merge_and_update(a, B)

# print("\nAfter update:")
# print(f"a.b.name: {a.b.name}")
# print(f"a.c: {a.c}")



import inspect

class B:
    name = "Original B"
    def __init__(self):
        self.other_attr = "Original other_attr"

class A:
    def __init__(self):
        self.b: B = B()
        self.c = self.b.name
        self.d = self.b.other_attr

# Function to dynamically update attributes of the instance based on the new class type
def merge_and_update(instance, new_class):
    # First, update the attributes that are instances of the old class
    for key, value in instance.__dict__.items():
        if isinstance(value, type(instance.b)):
            setattr(instance, key, new_class())
    
    # Next, re-evaluate any attributes that depend on updated attributes
    for key, value in instance.__dict__.items():
        # Check if the attribute's value depends on another attribute
        if isinstance(value, str):
            for inner_key, inner_value in instance.__dict__.items():
                if isinstance(inner_value, new_class):
                    # Dynamically evaluate if the value contains an attribute reference
                    for attr in dir(inner_value):
                        if not attr.startswith('__') and not callable(getattr(inner_value, attr)):
                            # Update the dependent attribute
                            if getattr(inner_value, attr) == value:
                                setattr(instance, key, getattr(inner_value, attr))

# Base instance
a = A()
print("Before update:")
print(f"a.b.name: {a.b.name}")
print(f"a.c: {a.c}")
print(f"a.d: {a.d}")

# New class definition
class B:
    name = "Updated B"
    def __init__(self):
        self.other_attr = "Updated other_attr"

# Update and reinitialize
merge_and_update(a, B)

print("\nAfter update:")
print(f"a.b.name: {a.b.name}")
print(f"a.c: {a.c}")
print(f"a.d: {a.d}")
