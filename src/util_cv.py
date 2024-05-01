import os
import pandas as pd
import numpy as np
def import_structure(filename):
    input_tmp = []
    with open(filename) as inputfile:
        for i, line in enumerate(inputfile):
            input_tmp.append(line)

    features = []
    modules = []
    output = []

    for line in input_tmp:
        line = line.rstrip()
        line = line.split(",")
        key = line[0]
        values = line[1:]
        if key == "F":
            features = values
        if (key == "M") or (key == "O"):
            modules.append([values[0], values[1:]])
        if key == "O":
            output = values[0]


    return features, modules, output

def import_data(filename):
    input_tmp = []
    with open(filename) as inputfile:
        for i, line in enumerate(inputfile):
            input_tmp.append(line)

    data = {}
    colnames = input_tmp[0].split(",")
    for line in input_tmp[1:]:
        line = line.rstrip()
        line = line.split(",")
        module_name = line[0]
        values = line[1:]

        data[module_name] = []
        for item in values:
            try:
                data[module_name].append(float(item))
            except ValueError:
                data[module_name].append(item)

    return data, colnames

def cal_depth_module(modules, key_output, features):
    queue = [key_output]
    modules_depth = {key_output: 0}
    while len(queue) != 0:
        q_pop = queue.pop(0)

        for module_input in modules[q_pop]:
            if module_input in features:
                continue
            if module_input not in modules_depth:
                modules_depth[module_input] = modules_depth[q_pop] + 1
            else:
                if modules_depth[module_input] > modules_depth[q_pop] + 1:  # kürzere depth nehmen, um kürzeste Tiefe bei Schleife zu nehmen, kommt im Moment nicht vor
                    modules_depth[module_input] = modules_depth[q_pop] + 1

            queue.append(module_input)

    depth_max = max(list(modules_depth.values()))
    for f in features:
        modules_depth[f] = depth_max + 1

    return modules_depth

def import_pw_names(path):
    delim = ","
    names = {}
    with open(file=path) as f:
        for row in f:
            row = row.rstrip().rsplit("\t")
            for e in range(3):
                name = row[e].split(delim)
                names[name[1]] = name[2]

            name_ilmn = row[-1].split(delim)[1]
            name_kegg = row[-2].split(delim)[2]
            names[name_ilmn] = name_kegg
    return names

def import_ilmn_hsa(path):
    delim = ","
    names = {}
    with open(file=path) as f:
        for row in f:
            row = row.rstrip().rsplit("\t")
            name_ilmn = row[-1].split(delim)[1]
            name_kegg = row[-2].split(delim)[1]
            names[name_ilmn] = name_kegg
    return names

def create_directory(directory_new):
    current_path = os.getcwd()
    try:
        os.mkdir(current_path + "/" + directory_new)
    except OSError:
        print("Creation of the directory %s failed" % current_path + directory_new)
    else:
        print("Successfully created the directory %s " % current_path + directory_new)

def import_best_model_values(path):
    dict_v = {}
    with open(file=path) as f:
        header = None
        for r in f:
            if header == None:
                header = r.split(sep=",")[:-1]
                continue
            values = r.split(sep=",")#[:-1]
            if values[0] == "ep":
                dict_v[values[0]] = [float(values[1]), float(values[1]), float(values[1])]  # epoche des besten models
            else:
                dict_v[values[0]] = [float(v) for v in values[1:-1]]

    return dict_v, header

def import_metric_values(path):
    data = []
    with open(file=path) as f:
        for r in f:
            data.append(float(r))
    return data

def import_pred(path):
    y_true = []
    y_pred = []
    with open(file=path) as f:
        for r in f:
            r = r.rstrip()
            r = r.replace("]", "")
            r = r.replace("[", "")
            y_true_tmp, y_pred_tmp = r.split(",")
            y_true_tmp = y_true_tmp.split()
            y_true_tmp = [float(v) for v in y_true_tmp]
            y_true.append(y_true_tmp)
            y_pred_tmp = y_pred_tmp.split()
            y_pred_tmp = [float(v) for v in y_pred_tmp]
            y_pred.append(y_pred_tmp)

    return [y_true, y_pred]

def import_conMat(path):
    data = []
    with open(file=path) as f:
        next(f)
        for r in f:
            r = r.rstrip()
            r = r.replace("]", "")
            r = r.replace("[", "")
            r = r.split()
            data.append([float(v) for v in r])
    return data

def import_lrp(path):
    with open(file=path) as f:
        header = None
        v_dict = {}
        for r in f:
            if r[-2] != ",": #TODO: Notlösung, "," am Ende fehlt in einigen Dateien
                if header == None:
                    header = r.split(sep=",")
                    header[-1] = header[-1].rstrip()
                    continue
                values = r.split(sep=",")
                v_dict[values[0]] = [float(v) for v in values[1:]]
            else:
                if header == None:
                    header = r.split(sep=",")[:-1]
                    continue
                values = r.split(sep=",")[:-1]
                v_dict[values[0]] = [float(v) for v in values[1:]]
    return v_dict, header

def combine_results(df, method):
    header = [f"mean_{method}", f"median_{method}", f"std_{method}",  f"min_{method}",  f"max_{method}"]
    results = {}
    for i, key_c in enumerate(df.index): #TODO: was tun, wenn alle Bewertungen 0 sind?
        results[key_c] = []
        results[key_c].append(np.mean(df.loc[key_c]))
        test = np.median(df.loc[key_c])
        if np.isnan(test):
            results[key_c].append(np.float64(0))
        else:
            results[key_c].append(np.median(df.loc[key_c]))
        results[key_c].append(np.std(df.loc[key_c]))
        results[key_c].append(np.min(df.loc[key_c]))
        results[key_c].append(np.max(df.loc[key_c]))
    df_new = pd.DataFrame.from_dict(results, orient='index')
    df_new.columns = header
    return df_new

def name_short(x):
    x = x.split(";")
    return x[0]