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
                            
                            _results = models.get(model_name, {})
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
                            
                            max_rows = len(models.get(model_name, {}))
                            for row_i in range(max_rows):
                                model_data = models.get(model_name, {})[row_i]
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

            # Read header row first
            headers = next(reader)  # Get column names
            headers = [h.strip() for h in headers]  # Clean spaces

            dataset_results = defaultdict(list)

            for row in reader:
                row = [cell.strip() if cell else "-" for cell in row]  # Replace empty cells with "-"

                if not any(row):  # Skip empty rows
                    continue

                # Create a dictionary mapping headers to row values
                row_dict = {headers[i]: row[i] for i in range(len(row))}

                model_key = row_dict.get(headers[0], "-")  # Use first column as key
                dataset_results[model_key].append(row_dict)

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
    tabular_format = "l|"  # "Models" and "Backbone"
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
    colspan = 3
    header1 = "    \\multicolumn{1}{c}{}"
    for i in range(num_datasets):
        dataset_name = datasets[i]
        header1 += f" & \\multicolumn{{{colspan}}}{{c}}{{\\textit{{{dataset_name}}}}}"

    total_cols = len([i for i in columns if i != '|']) + 2
    _colspan = total_cols - 2 - num_datasets * colspan
    header1 += f" & \\multicolumn{{{_colspan}}}{{c}}{{}}"
    header1 += " \\\\\n"
    header1 += "    \\specialrule{0.75pt}{0pt}{0pt}\n"


    # Second header row: column names
    header2 = "    Models"
    for col in columns:
        header2 += f" & {col[0]}"
    header2 += " \\\\\n    \\hline\n"

    latex_str += header1 + header2


    # Print rows for each model group
    total_cols = 2 + len(datasets) * 2 + sum(1 for col in columns if not col[3])  # Two columns per dataset
    group_idx = 0
    for group_name, group in model_groups.items():
        if len(group_name):
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

                _results = results[group_idx].get(datasets[0], {}).get(model_name, {})#.get(backbone_name, [{}])
                if idx is not None:
                    _results = [_results[idx]]

                max_rows = len(_results)
                for row_i in range(max_rows):
                    # if model_name == 'IAUNet':
                    #     latex_str += "\\rowcolor{our_results_color} \n"
                    #     _model_name = "\\textbf{IAUNet} (ours)"
                        
                    row_str = f"{_model_name}"

                    for col in columns:
                        _results = results[group_idx].get(datasets[col[4]], {}).get(model_name, {})#.get(backbone_name, [{}])
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
#     "": "temp/latex/[IAUNet] _ Benchmarks v2 - Experiments.csv",
#     # "": "temp/latex/[IAUNet] _ Benchmarks v2 - Ablations - PixelDecoder - v2.csv",
# }


# model_groups = {
#     '': [
#         [
#             ('v1', ''), ('v2', ''), ('v3', ''), ('v4', ''),
#             ('v5', ''), ('v6', ''), ('v7', ''), ('v8', ''),
#         ]
#     ]
# }

# columns = [
#     ['AP', 'mAP@0.5:0.95', 'p{0.4cm}', True, 0],
#     ['AP$_{50}$', 'mAP@0.5', 'p{0.4cm}', True, 0],
#     ['AP$_{75}$', 'mAP@0.75', 'p{0.4cm}|', True, 0],
#     ['\#params.', 'Params', 'c|', False],
#     ['FLOPs', 'FLOPs', 'c', False]
# ]

# metrics = ['mAP@0.5:0.95', 'mAP@0.5', 'mAP@0.75']

# model_renaming = {
#     'v1': '\\textbf{IAUNet (R50)}',
#     'v2': '\hspace{0.5em}+ mask branch',
#     'v3': '\hspace{0.5em}+ ffn (2048 $\\rightarrow$ 1024)',
#     'v4': '\hspace{0.5em}+ se block',
#     'v5': '\hspace{0.5em}+ coords',
#     'v6': '\hspace{0.5em}+ dec\_layers (1 $\\rightarrow$ 3) (seq.)',
#     'v7': '\hspace{0.5em}+ dec\_layers (1 $\\rightarrow$ 3) (stacked.)',
#     'v8': '\hspace{0.5em}+ deep\_supervision',
# }

# backbone_renaming = {
#     'R50': 'R50',
#     'R101': 'R101',
#     'swin-s': 'Swin-S',
#     'swin-b': 'Swin-B',
#     '-': ' '
# }

# dataset_dict = parse_csv_to_dict(datasets)
# pprint(dataset_dict)
# dataset_dicts = preprocess(dataset_dict, model_groups, metrics)
# pprint(dataset_dicts)
# latex_table = generate_latex_table(dataset_dicts, model_groups, columns, model_renaming=model_renaming, backbone_renaming=backbone_renaming)
# print(latex_table)






datasets = {
    # "": "temp/latex/[IAUNet] _ Benchmarks v2 - Experiments.csv",
    # "": "temp/latex/[IAUNet] _ Benchmarks v2 - Ablations - PixelDecoder - v2.csv",
    "": "temp/latex/[IAUNet] _ Benchmarks v2 - Ablations - num_queries.csv",
}


model_groups = {
    '': [
        [
            ('v9', ''), ('v10', ''), ('v11', ''), ('v12', ''),
        ]
    ]
}

columns = [
    ['num\_queries', 'num_queries', 'c|', False, 0],
    ['AP', 'mAP@0.5:0.95', 'p{0.4cm}', True, 0],
    ['AP$_{50}$', 'mAP@0.5', 'p{0.4cm}', True, 0],
    ['AP$_{75}$', 'mAP@0.75', 'p{0.4cm}|', True, 0],
    ['FLOPs', 'FLOPs', 'c', False, 0]
]

metrics = ['mAP@0.5:0.95', 'mAP@0.5', 'mAP@0.75']

# model_renaming = {
#     'v9': '\hspace{0.5em}+ full skip',
#     'v10': '\hspace{0.5em}+ $1 \\times 1$ skip concat',
#     'v11': '\hspace{0.5em}+ $1 \\times 1$ skip add',
#     'v12': '\hspace{0.5em}+ light mask head',
# }

model_renaming = {
    'v9': '\hspace{0.5em}+ full skip',
    'v10': '\hspace{0.5em}+ $1 \\times 1$ skip concat',
    'v11': '\hspace{0.5em}+ $1 \\times 1$ skip add',
    'v12': '\hspace{0.5em}+ light mask head',
}

backbone_renaming = {
    'R50': 'R50',
    'R101': 'R101',
    'swin-s': 'Swin-S',
    'swin-b': 'Swin-B',
    '-': ' '
}

dataset_dict = parse_csv_to_dict(datasets)
pprint(dataset_dict)
dataset_dicts = preprocess(dataset_dict, model_groups, metrics)
# pprint(dataset_dicts)
latex_table = generate_latex_table(dataset_dicts, model_groups, columns, model_renaming=model_renaming, backbone_renaming=backbone_renaming)
print(latex_table)
