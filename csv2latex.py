import csv
from collections import defaultdict
from copy import deepcopy



def preprocess(dataset_dict, model_groups, metrics):
    # we will return a list of dictionaries for each model group.
    # the format of the dictionary will be the same as the input dataset_dict.
    # we will update the values of the metrics with best, second best, or '-'.
    results = []

    for group_name, group in model_groups.items():
        for group_models in group:

            group_dict = deepcopy(dataset_dict)
            for dataset_name, models in group_dict.items():

                for metric in metrics:
                        metric_values = []

                        for model in group_models:
                            if len(model) == 3:
                                model_name, backbone_name, idx = model
                            else:
                                model_name, backbone_name = model
                                idx = None
                            
                            _results = models.get(model_name, {}).get(backbone_name, [{}])
                            if idx is not None:
                                _results = [_results[idx]]
                            max_rows = len(_results)
                            for row_i in range(max_rows):
                                model_data = _results[row_i]
                                metric_value = model_data.get(metric, '--')
                                metric_values.append((model_name, backbone_name, float(metric_value)))

                        # sort.
                        metric_values.sort(key=lambda x: x[2], reverse=True)

                        # get best and second best values.
                        if len(metric_values) > 0:
                            best_value = metric_values[0][2]
                            second_best_value = metric_values[1][2] if len(metric_values) > 1 else -1
                            i = 2
                            while second_best_value == best_value:
                                second_best_value = metric_values[i][2]
                                i += 1


                        # update the model data with best, second best, or '-'.
                        for model in group_models:
                            if len(model) == 3:
                                model_name, backbone_name, idx = model
                            else:
                                model_name, backbone_name = model
                                idx = None
                            
                            max_rows = len(models.get(model_name, {}).get(backbone_name, [{}]))
                            for row_i in range(max_rows):
                                model_data = models.get(model_name, {}).get(backbone_name, [{}])[row_i]
                                current_value = model_data.get(metric, '--')

                                if current_value == '--':
                                    model_data[metric] = '-'
                                elif float(current_value) == best_value:
                                    model_data[metric] = (current_value, 'best')
                                elif float(current_value) == second_best_value:
                                    model_data[metric] = (current_value, 'second best')
                                else:
                                    model_data[metric] = (current_value, '-')

            results.append(group_dict)
        
    return results


def parse_csv_to_dict(datasets):
    """
    Parses multiple dataset CSVs into a structured dictionary.
    
    :param datasets: Dictionary mapping dataset names to CSV file paths.
    :return: Dictionary containing parsed data.
    """
    results = {}

    for dataset, filepath in datasets.items():
        with open(filepath, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            headers = None
            dataset_results = defaultdict(lambda: defaultdict(list))
            current_model = None

            for row in reader:
                row = [cell.strip() for cell in row]  # Clean spaces

                if not any(row):  # Skip empty rows
                    continue

                if row[0] == "num_queries":  # Identify repeated headers
                    headers = row
                    continue

                model_info = row[0]
                num_queries = row[1]

                if "(" in model_info and ")" in model_info:  # Extract model name and backbone
                    model_name, backbone = model_info.split("(", 1)
                    backbone = backbone.rstrip(")")
                else:
                    model_name, backbone = model_info, None

                if backbone is None:
                    backbone = '-'

                model_key = model_name.strip()

                if not len(model_key):
                    continue


                dataset_results[model_key][backbone].append({
                    "num_queries": num_queries,
                    "mAP@0.5:0.95": row[2],
                    "mAP@0.5": row[3],
                    "mAP@0.75": row[4],
                    "mAP(s)": row[5],
                    "mAP(m)": row[6],
                    "mAP(l)": row[7],
                    "Params": row[8],
                    "FLOPs": row[9] if len(row) > 9 else "-"
                })

        results[dataset] = dataset_results

    return results



from collections import defaultdict
from pprint import pprint

from collections import defaultdict
from pprint import pprint


def preprocess_columns(columns, datasets):
    column_counts = {}
    for col in columns:
        col_name = col[0]

        if col_name not in column_counts:
            column_counts[col_name] = 0
        
        # check if the column already has a dataset id.
        if len(col) == 5:
            continue
        col.append(column_counts[col_name])
        column_counts[col_name] += 1

    return columns


def generate_latex_table(results, model_groups, columns, model_renaming={}, backbone_renaming={}):
    """
    Generates a LaTeX table string from results, ensuring dataset-specific columns are correctly repeated
    while maintaining the original order of columns.
    """
    datasets = list(results[0].keys())

    # Preprocess the columns to duplicate dataset-specific columns
    # we need to add extra variable to columns to indicate the dataset id
    # of repeating columns we increment the value
    columns = preprocess_columns(columns, datasets)

    # Define LaTeX tabular format
    tabular_format = "l|c|"  # "Models" and "Backbone"
    for col in columns:
        tabular_format += col[2]  # Add the column format

    # Start table
    latex_str = f"""
\\renewcommand{{\\arraystretch}}{{1.2}}
\\begin{{table*}}[!t]
\\vspace{{-0.35cm}}
\\centering
\\scriptsize
\\begin{{tabular}}{{{tabular_format}}}
"""

    # First header row: dataset names
    num_datasets = len(datasets)
    colspan = 2
    header1 = "    \\multicolumn{3}{c}{}"
    for i in range(num_datasets):
        dataset_name = datasets[i]
        header1 += f" & \\multicolumn{{{colspan}}}{{c}}{{\\textit{{{dataset_name}}}}}"

    total_cols = len([i for i in columns if i != '|']) + 2
    _colspan = total_cols - 3 - num_datasets * colspan
    header1 += f" & \\multicolumn{{{_colspan}}}{{c}}{{}}"
    header1 += " \\\\\n"
    header1 += "    \\specialrule{0.75pt}{0pt}{0pt}\n"


    # Second header row: column names
    header2 = "    Models & backbones"
    for col in columns:
        header2 += f" & {col[0]}"
    header2 += " \\\\\n    \\hline\n"

    latex_str += header1 + header2


    # Print rows for each model group
    total_cols = 2 + len(datasets) * 2 + sum(1 for col in columns if not col[3])  # Two columns per dataset
    group_idx = 0
    for group_name, group in model_groups.items():
        latex_str += f"\\multicolumn{{{total_cols}}}{{l}}{{\\textit{{\\textbf{{{group_name}}}}}}} \\\\\n\\hline\n"

        for group_models in group:
            for model in group_models:
                # model: ('MaskRCNN', 'R50') or ('MaskRCNN', 'R50', 0)
                # this means if the model has multiple values we take the idx-th value.
                if len(model) == 3:
                    model_name, backbone_name, idx = model
                else:
                    model_name, backbone_name = model
                    idx = None

                _model_name = model_renaming.get(model_name, model_name)
                _backbone_name = backbone_renaming.get(backbone_name, backbone_name)

                _results = results[group_idx].get(datasets[0], {}).get(model_name, {}).get(backbone_name, [{}])
                if idx is not None:
                    _results = [_results[idx]]

                max_rows = len(_results)
                for row_i in range(max_rows):
                    # if model_name == 'IAUNet':
                    #     latex_str += "\\rowcolor{our_results_color} \n"
                    #     _model_name = "\\textbf{IAUNet} (ours)"
                        
                    row_str = f"{_model_name} & {_backbone_name}"

                    for col in columns:
                        _results = results[group_idx].get(datasets[col[4]], {}).get(model_name, {}).get(backbone_name, [{}])
                        if idx is not None:
                            _results = [_results[idx]]

                        value = _results[row_i].get(col[1], '--')
                        
                        # check for best and second best values.
                        # value: ('44.7', 'second best') or value: '44.7'
                        if isinstance(value, tuple):
                            if value[1] == 'best':
                                value = f"\\textbf{{{value[0]}}}"
                            elif value[1] == 'second best':
                                value = f"\\underline{{{value[0]}}}"
                            else:
                                value = value[0]

                        row_str += f" & {value}"

                    latex_str += row_str + " \\\\\n"
            
            group_idx += 1
            latex_str += "\\hline \n\n"

    latex_str += """
 \\end{tabular}
 \\caption{}
 \\label{table:metrics_all}
 \\vspace{-0.3cm}
\\end{table*}
"""
    return latex_str






from pprint import pprint

# datasets = {
#     "LiveCell": "temp/latex/[IAUNet] _ Benchmarks v2 - LiveCellCrop.csv",
#     "EVICAN2$_E$": "temp/latex/[IAUNet] _ Benchmarks v2 - EVICAN2_Easy.csv",
#     "EVICAN2$_M$": "temp/latex/[IAUNet] _ Benchmarks v2 - EVICAN2_Medium.csv",
#     "EVICAN2$_D$": "temp/latex/[IAUNet] _ Benchmarks v2 - EVICAN2_Difficult.csv",
#     "ISBI2014": "temp/latex/[IAUNet] _ Benchmarks v2 - ISBI2014.csv",
# }

# model_groups = {
#     'Models with Convolution-Based Backbones': [
#         [
#             ('MaskRCNN', 'R50'),
#             ('PointRend', 'R50'),
#             ('Mask2Former', 'R50'),
#             ('MaskDino', 'R50'),
#             ('IAUNet', 'R50'),

#         ],
#         [
#             ('MaskRCNN', 'R101'),
#             ('PointRend', 'R101'),
#             ('Mask2Former', 'R101'),
#             ('MaskDino', 'R101'),
#             ('IAUNet', 'R101'),
#         ]
#     ],

#     'Models with Transformer-Based Backbones': [
#         [
#             ('MaskRCNN', 'swin-s'),
#             ('PointRend', 'swin-s'),
#             ('Mask2Former', 'swin-s'),
#             ('MaskDino', 'swin-s'),
#             ('IAUNet', 'swin-s'),
#         ], 
#         [
#             ('MaskRCNN', 'swin-b'),
#             ('PointRend', 'swin-b'),
#             ('Mask2Former', 'swin-b'),
#             ('MaskDino', 'swin-b'),
#             ('IAUNet', 'swin-b'),
#         ]
#     ], 

#     'Specialized Cell Segmentation Methods': [
#         [
#             ('cellpose', '-'),
#             ('cellpose + sm', '-'),
#             ('CellDETR', 'R34'),
#             ('IAUNet', 'R50', 0),
#         ]
#     ],

#     'YOLO Family': [
#         [
#             ('yolov8-m', '-'),
#             ('yolov8-l', '-'),
#             ('yolov8-x', '-'),
#             ('IAUNet', 'swin-s', 0),
#         ],
#         [
#             ('yolov9-e', '-'),
#             ('yolov9-c', '-'),
#             ('IAUNet', 'swin-s', 0),
#         ]
#     ],

#     'SAM Family': [
#         [
#             ("sam-b ['points']", '-'),
#             ("sam-b ['bboxes']", '-'),
#             ('IAUNet', 'swin-s', 0),
#         ], 
#         [
#             ("sam-l ['points']", '-'),
#             ("sam-l ['bboxes']", '-'),
#             ('IAUNet', 'swin-b', 1),
#         ]
#     ],
# }


# columns = [
#     ['num\_queries', 'num_queries', 'c|', False],
#     ['AP', 'mAP@0.5:0.95', 'p{0.4cm}', True, 0],
#     ['AP$_{50}$', 'mAP@0.5', 'p{0.4cm}|', True, 0],
#     ['AP', 'mAP@0.5:0.95', 'p{0.4cm}', True, 1],
#     ['AP$_{50}$', 'mAP@0.5', 'p{0.4cm}|', True, 1],
#     ['AP', 'mAP@0.5:0.95', 'p{0.4cm}', True, 2],
#     ['AP$_{50}$', 'mAP@0.5', 'p{0.4cm}|', True, 2],
#     ['AP', 'mAP@0.5:0.95', 'p{0.4cm}', True, 3],
#     ['AP$_{50}$', 'mAP@0.5', 'p{0.4cm}|', True, 3],
#     ['AP', 'mAP@0.5:0.95', 'p{0.4cm}', True, 4],
#     ['AP$_{50}$', 'mAP@0.5', 'p{0.4cm}|', True, 4],
#     ['\#params.', 'Params', 'c|', False],
#     ['FLOPs', 'FLOPs', 'c', False]
# ]

# metrics = ['mAP@0.5:0.95', 'mAP@0.5', 'mAP@0.75']







datasets = {
    "Revvity-25": "temp/latex/[IAUNet] _ Benchmarks v2 - Revvity-25.csv",
}

# model_groups = {
#     'Models with Convolution-Based Backbones': [
#         [
#             ('MaskRCNN', 'R50'),
#             ('PointRend', 'R50'),
#             ('Mask2Former', 'R50'),
#             ('MaskDino', 'R50'),
#             ('IAUNet', 'R50'),

#         ],
#         [
#             ('MaskRCNN', 'R101'),
#             ('PointRend', 'R101'),
#             ('Mask2Former', 'R101'),
#             ('MaskDino', 'R101'),
#             ('IAUNet', 'R101'),
#         ]
#     ],

#     'Models with Transformer-Based Backbones': [
#         [
#             ('MaskRCNN', 'swin-s'),
#             ('PointRend', 'swin-s'),
#             ('Mask2Former', 'swin-s'),
#             ('MaskDino', 'swin-s'),
#             ('IAUNet', 'swin-s'),
#         ], 
#         [
#             ('MaskRCNN', 'swin-b'),
#             ('PointRend', 'swin-b'),
#             ('Mask2Former', 'swin-b'),
#             ('MaskDino', 'swin-b'),
#             ('IAUNet', 'swin-b'),
#         ]
#     ], 
# }

model_groups = {
    'YOLO Family': [
        [
            ('yolov8-m', '-'),
            ('yolov8-l', '-'),
            ('yolov8-x', '-'),
            ('IAUNet', 'swin-s', 0),
        ],
        [
            ('yolov9-e', '-'),
            ('yolov9-c', '-'),
            ('IAUNet', 'swin-s', 0),
        ]
    ],

    'SAM Family': [
        [
            ("sam-b ['points']", '-'),
            ("sam-b ['bboxes']", '-'),
            ('IAUNet', 'swin-s', 0),
        ], 
        [
            ("sam-l ['points']", '-'),
            ("sam-l ['bboxes']", '-'),
            ('IAUNet', 'swin-b', 1),
        ]
    ],
}


columns = [
    ['num\_queries', 'num_queries', 'c|', False],
    ['AP', 'mAP@0.5:0.95', 'p{0.4cm}', True, 0],
    ['AP$_{50}$', 'mAP@0.5', 'p{0.4cm}', True, 0],
    ['AP$_{75}$', 'mAP@0.75', 'p{0.4cm}|', True, 0],
    ['AP$_{S}$', 'mAP(s)', 'p{0.4cm}', True, 0],
    ['AP$_{M}$', 'mAP(m)', 'p{0.4cm}', True, 0],
    ['AP$_{L}$', 'mAP(l)', 'p{0.4cm}|', True, 0],
    ['\#params.', 'Params', 'c|', False],
    ['FLOPs', 'FLOPs', 'c', False]
]

metrics = ['mAP@0.5:0.95', 'mAP@0.5', 'mAP@0.75', 'mAP(s)', 'mAP(m)', 'mAP(l)']




model_renaming = {
    'MaskRCNN': 'Mask R-CNN \cite{mask_rcnn}',
    'PointRend': 'PointRend \cite{pointrend}',
    'Mask2Former': 'Mask2Former \cite{mask2former}',
    'MaskDino': 'MaskDINO \cite{mask_dino}',
    
    'IAUNet': '\\rowcolor{our_results_color} \n\\textbf{IAUNet (ours)}',

    'CellDETR': 'CellDETR \cite{cell_detr}',
    'cellpose': 'CellPose \cite{cellpose}',
    'cellpose + sm': 'CellPose + SM \cite{cellpose2}',
    
    'yolov8-m': 'YOLOv8-M \cite{yolov8}',
    'yolov8-l': 'YOLOv8-L \cite{yolov8}',
    'yolov8-x': 'YOLOv8-X \cite{yolov8}',
    'yolov9-e': 'YOLOv9-E \cite{yolov9}',
    'yolov9-c': 'YOLOv9-C \cite{yolov9}',
    
    "sam-b ['points']": 'SAM-B \\textit{(points)} \cite{sam}',
    "sam-b ['bboxes']": 'SAM-B \\textit{(boxes)} \cite{sam}',
    "sam-l ['points']": 'SAM-L \\textit{(points)} \cite{sam}',
    "sam-l ['bboxes']": 'SAM-L \\textit{(boxes)} \cite{sam}',
}

backbone_renaming = {
    'R50': 'R50',
    'R101': 'R101',
    'swin-s': 'Swin-S',
    'swin-b': 'Swin-B',
    '-': ' '
}

dataset_dict = parse_csv_to_dict(datasets)
dataset_dicts = preprocess(dataset_dict, model_groups, metrics)
latex_table = generate_latex_table(dataset_dicts, model_groups, columns, model_renaming=model_renaming, backbone_renaming=backbone_renaming)
print(latex_table)
