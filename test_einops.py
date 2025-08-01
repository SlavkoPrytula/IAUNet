# import os

# def print_dir_tree(root_dir, indent=""):
#     """Recursively prints the directory tree structure."""
#     try:
#         entries = sorted(os.listdir(root_dir))
#     except PermissionError:
#         print(f"{indent}[Permission Denied] {root_dir}")
#         return
    
#     print(entries)
#     raise
#     for index, entry in enumerate(entries):
#         if 'runs' not in entry:
#             path = os.path.join(root_dir, entry)
#             is_last = index == len(entries) - 1
#             prefix = "└── " if is_last else "├── "
            
#             print(indent + prefix + entry)
            
#             if os.path.isdir(path):
#                 new_indent = indent + ("    " if is_last else "│   ")
#                 print_dir_tree(path, new_indent)


# root_path = "./"
# print_dir_tree(root_path)



# # data_dir = "/gpfs/helios/home/etais/com_palo/2025-03-06_OrganoidExport1"



data_root = "/gpfs/space/projects/PerkinElmer/cytoplasm_segmentation/datasets/Revvity-25/v2/annotations/valid.json"

# annotations/valid.json


# lets print out the files of the json 

import json

with open(data_root, 'r') as f:
    data = json.load(f)

# print the keys of the json
print("Keys in the JSON file:")
for key in data.keys():
    print(f"- {key}")


# data['info'] = {
#     "description": "Revvity-25 dataset for cytoplasm segmentation",
#     "url": "https://www.revvity.com",
#     "version": "1.0",
#     "year": 2025,
#     "contributor": "Revvity",
#     "date_created": "2025-01-01"
# }

# # and save it back to the file
# with open(data_root, 'w') as f:
#     json.dump(data, f, indent=4)