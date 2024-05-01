import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sn
import numpy as np
from operator import itemgetter
import pandas as pd

import torch
from src import LRP
#import LRP
from sys import getsizeof

from sklearn.metrics import confusion_matrix
from sklearn import metrics

def import_structure(filename):
    input_tmp = []
    with open(filename) as inputfile:
        for i, line in enumerate(inputfile):
            input_tmp.append(line)

    features = []
    #labels = []
    neurons = []
    #modules = []
    output = []

    for line in input_tmp:
        line = line.rstrip()
        line = line.split(",")
        key = line[0]
        values = line[1:]
        if key == "L":
            labels = values#info.split(",")
        if key == "F":
            features = values#info.split(",")
        if (key == "M") or (key == "O"):
            #info = info.split(",")
            neurons.append([values[0], values[1:]])
        if key == "O":
            #info = info.split(",")
            #output.append(info[0])
            output = values[0]
        """if pre == "M":
            info = info.split(";")
            modules.append([info[0].split(",")[0], info[1].split(","), info[2].split(",")])"""

    return features, neurons, output#, modules

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
            try: #TODO: In Dataset auslagern?
                data[module_name].append(float(item))
            except ValueError:
                data[module_name].append(item)

    return data, colnames

def create_directory(directory_new):
    current_path = os.getcwd()
    try:
        os.mkdir(current_path + "/" + directory_new)
    except OSError:
        print("Creation of the directory %s failed " % current_path + directory_new)
    else:
        print("Successfully created the directory %s " % current_path + directory_new)

def notes_to_file(path, notes, dicts):
    file = open(path + "/notes.txt", "w")
    file.write(notes)
    file.write("\n")
    for dict_tmp in dicts:
        for key, item in dict_tmp.items():
            file.write(key + ": " + str(item))
            file.write("\n")
    file.close()

#TODO: auf neue importance anpassen, brauche ich aber gerade nicht.
def plot_importance(importance, module_order, filename):
    labels, data = importance.keys(), importance.values()
    importance_abs = {key: [abs(ele) for ele in val] for key, val in importance.items()}
    data = importance_abs.values()
    plt.figure(figsize=(20, 10))
    plt.boxplot(data, showfliers=False)
    plt.xticks(range(1, len(labels) + 1), labels)
    if(filename==None):
        plt.show()
    else:
        plt.savefig(str(filename) + ".png")
    plt.clf()
    plt.close()

def file_importance(importance, columnames, module_order, modules, features, path):
    create_directory(directory_new=path)
    depth_max = max([m[1] for m in module_order]) + 1
    depth_dict = {int(depth_max): features}
    for name, d in module_order:
        if d in depth_dict:
            depth_dict[d].append(name)
        else:
            depth_dict[d] = [name]

    for key, module in modules.items():
        imp_tmp = None
        for key_neuron in module["hidden_layer"][0]:
            if imp_tmp == None:
                imp_tmp = torch.Tensor(importance[key_neuron])
                imp_res_tmp = torch.Tensor(importance[f"{key_neuron}_residual"])
            else:
                imp_tmp = torch.add(imp_tmp, torch.Tensor(importance[key_neuron]))
                imp_res_tmp = torch.add(imp_res_tmp, torch.Tensor(importance[f"{key_neuron}_residual"]))
        importance[key] = imp_tmp
        importance[f"{key}_residual"] = imp_res_tmp

    for d, key_modules in depth_dict.items():
        with open(f"{path}/depth_{d}.csv", "w") as f:
            f.write(f"Module,")
            for c in columnames:
                f.write(f"{c},")
            f.write("\n")
            for key_module in key_modules:
                f.write(f"{key_module},")
                for v in importance[key_module]:
                    f.write(f"{v.detach().numpy()},")
                f.write("\n")

                f.write(f"{key_module}_residual,")
                for v in importance[f"{key_module}_residual"]:
                    f.write(f"{v.detach().numpy()},")
                f.write("\n")

def mean_reduction(values):
    mean_v = np.mean(values)
    new_values = []
    for item in values:
        new_values.append(float(item - mean_v))
    return new_values

def evaluate_MNN(result, path, c):
    # Error as Plot
    result_keys = ["acc", "f1", "loss", "sens", "spec"]
    # for key, item in result.items():
    for key in result_keys:
        item = result[key]
        plt.title(key)
        plt.figure(figsize=(20, 10))
        plt.plot(item["train"], color="k", label="train")
        plt.plot(item["val"], color="g", label="validation")
        plt.plot(item["test"], color="b", label="test")
        plt.legend()
        plt.savefig(fname=path + "/" + key)
        plt.close()

        for key_dataset, item_dataset in item.items():
            with open(path + "/" + key + "_" + key_dataset + ".txt", "w") as file:
                for value in item_dataset:
                    file.write(str(value))
                    file.write("\n")

    # Grafik, wie sicher sich das Netz für die Vorhersagen einzelner Klassen ist.
    # Eine Farbe -> Eine Klasse
    # y-Achse: Vorhersage für bekanntes Label
    for key, pred_dataset_tmp in result["y_pred_softmax"].items():
        if (len(pred_dataset_tmp) == 0):  # Wenn keine Testdaten genutzt werden.
            continue
        pred_dataset_tmp = pred_dataset_tmp.detach().numpy()
        label_len = len(pred_dataset_tmp[0])
        label_pred_softmax = [[] for _ in range(label_len)]

        # Finden wie die Vorhersage für das richtige Label ist
        y_true_tmp = result["y_true"][key].detach().numpy()
        for pos, pred_single in enumerate(pred_dataset_tmp):
            y_true_index = list(y_true_tmp[pos]).index(max(y_true_tmp[pos]))
            y_pred_tmp = pred_single[y_true_index]
            label_pred_softmax[y_true_index].append(y_pred_tmp)

        plt.title(key)
        plt.figure(figsize=(20, 10))
        plt.hist(label_pred_softmax, bins=100)  # ! Dichte für allle Vorhersagen wäre super
        plt.savefig(fname=path + "/outputs_" + key)
        plt.close()

        file = open(path + "/outputs_" + key + ".txt", "w")
        for i, item in enumerate(pred_dataset_tmp):
            file.write(str(y_true_tmp[i]))
            file.write(";")
            file.write(str(item))
            file.write("\n")
        file.close()

    # Eval und Plot von LRP
    #  Epsilon
    lrp = LRP.LRP(controler=c)
    args_lrp = {"epsilon": 0.1, }
    lrp.set_args(args=args_lrp)
    importance_nodes, _ = lrp.eval_dataset(dataset="train", type="lrp_epsilon")
    plot_importance(importance=importance_nodes, filename=path + "/" + "epsilon_train")
    file_importance(importance=importance_nodes, filename=path + "/" + "epsilon_train_values")
    print("Epsilon Train LRP done")

    lrp = LRP.LRP(controler=c)
    args_lrp = {"epsilon": 0.1, }
    lrp.set_args(args=args_lrp)
    importance_nodes, _ = lrp.eval_dataset(dataset="validation", type="lrp_epsilon")
    plot_importance(importance=importance_nodes, filename=path + "/" + "epsilon_val")
    file_importance(importance=importance_nodes, filename=path + "/" + "epsilon_val_values")
    print("Epsilon Val LRP done")

    """if (args_controler["test_size"] != 0):
        lrp = LRP.LRP(controler=c)
        args_lrp = {"epsilon": 0.1, }
        lrp.set_args(args=args_lrp)
        importance_nodes, _ = lrp.eval_dataset(dataset="test", type="lrp_epsilon")
        util.plot_importance(importance=importance_nodes, filename=path + "/" + "epsilon_test")
        util.file_importance(importance=importance_nodes, filename=path + "/" + "epsilon_test_values")
        print("Epsilon Test LRP done")"""

    """# Gamma
    lrp = LRP.LRP(controler=c)
    args_lrp = {"gamma": 2, }
    lrp.set_args(args=args_lrp)
    importance_nodes, _ = lrp.eval_dataset(dataset="train", type="lrp_gamma")
    plot_importance(importance=importance_nodes, filename=path + "/" + "gamma_train")
    file_importance(importance=importance_nodes, filename=path + "/" + "gamma_train_values")
    print("Epsilon Train LRP done")

    lrp = LRP.LRP(controler=c)
    args_lrp = {"gamma": 2, }
    lrp.set_args(args=args_lrp)
    importance_nodes, _ = lrp.eval_dataset(dataset="validation", type="lrp_gamma")
    plot_importance(importance=importance_nodes, filename=path + "/" + "gamma_val")
    file_importance(importance=importance_nodes, filename=path + "/" + "gamma_val_values")
    print("Epsilon Val LRP done")"""

    """if (args_controler["test_size"] != 0):
        lrp = LRP.LRP(controler=c)
        args_lrp = {"gamma": 2, }
        lrp.set_args(args=args_lrp)
        importance_nodes, _ = lrp.eval_dataset(dataset="test", type="lrp_gamma")
        util.plot_importance(importance=importance_nodes, filename=path + "/" + "gamma_test")
        util.file_importance(importance=importance_nodes, filename=path + "/" + "gamma_test_values")
        print("Epsilon Test LRP done")"""

def export_saved_grad(grad):
    None #TODO: hier export vorbereiten
    print("hi")
    values = {key: {} for key in grad["0"].keys()}
    list_ep = list(grad.keys())
    list_ep.sort()
    for ep in list_ep:
        for key_neuron in grad[ep].keys():
            for key_depth in grad[ep][key_neuron].keys():
                for pos, weight in enumerate(grad[ep][key_neuron][key_depth]):
                    if str(pos) not in values[key_neuron]:
                        values[key_neuron][str(pos)] = {} #TODO: was ist mit "iter passiert?

                        if str(key_depth) not in values[key_neuron][str(pos)]:
                            values[key_neuron][str(pos)][str(key_depth)] = [weight]
                        else:
                            values[key_neuron][str(pos)][str(key_depth)].append(weight)
                    else:
                        if str(key_depth) not in values[key_neuron][str(pos)]:
                            values[key_neuron][str(pos)][str(key_depth)] = [weight]
                        else:
                            values[key_neuron][str(pos)][str(key_depth)].append(weight)

    for key_neuron in values.keys():
        for key_weight in values[key_neuron].keys():
            for key_depth in values[key_neuron][key_weight].keys():
                plt.plot(values[key_neuron][key_weight][key_depth], color="b")
                plt.annotate(key_depth, (0, values[key_neuron][key_weight][key_depth][0]))
            plt.savefig(f"grad/{key_neuron}_{key_weight}.png")
            plt.close()
    #print(ep)

def export_module_error(module_error, module_error_ep, path):
    # TODO: Es werden noch keine Wert in einer Tabelle ausgegeben

    path_tmp = f"{path}/error"
    create_directory(directory_new=path_tmp)
    """lasterror = [module_error[key][-1] for key in module_error.keys()]
    plt.hist(lasterror)
    plt.savefig(f"{path_new}/lasterror.png")
    plt.close('all')"""

    key_lasterror = [[key, module_error[key][-1]] for key in module_error.keys()]
    key_lasterror.sort(key=lambda x: x[1])

    value_min = float("inf")
    value_max = float(0)
    for key, lasterror in key_lasterror:
        value_min_tmp = min(module_error[key])
        if value_min_tmp < value_min:
            value_min = value_min_tmp
        value_max_tmp = max(module_error[key])
        if value_max_tmp > value_max:
            value_max = value_max_tmp

        plt.plot(module_error_ep, module_error[key])
        plt.savefig(f"{path_tmp}/{key}_error.png")
        plt.close('all')
    value_min = value_min - 0.1
    value_max = value_max + 0.1

    i = 1
    k = 5
    for key, lasterror in key_lasterror:
        plt.plot(module_error[key], label=key)
        if i % k == 0:
            plt.ylim(value_min, value_max)
            plt.legend()
            plt.savefig(f"{path_tmp}/bulk_{i - k+1}_{i}.png")
            plt.close('all')
        i += 1
    else:
        if i % k != 0:
            plt.ylim(value_min, value_max)
            plt.legend()
            plt.savefig(f"{path_tmp}/bulk_{i - (i % k)}_{i-1}.png")
            plt.close('all')

import sys

# https://stackoverflow.com/questions/1331471/in-memory-size-of-a-python-structure
def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size

def export_model_structure(c, path):
    path_new = f"{path}/model_structure"
    create_directory(directory_new=path_new)
    with open(f"{path_new}/modules.txt", "w") as f:
        for key, module in c.module.items():
            f.write(key)
            f.write(",")
            f.write(str(module["depth"]))
            f.write(",")
            for n in module["neurons"]:
                f.write(n.name)
                f.write(",")
            f.write("\n")

def get_size_of_dict(dict_size):
    size = 0
    for value in dict_size.values():
        if isinstance(value, dict):
            size_tmp = get_size_of_dict(value)
        else:
            size_tmp = getsizeof(value)/1_000_000_000
        size += size_tmp
    return size

import time
def print_batch_computation_time(time_batch, i_batch, batch_size, len_dataset):
    time2 = time.time()
    if ((time2 - time_batch[1]) / 60) >= 1:
        text = f"\r \t time passed {round((time2 - time_batch[0]) / 60, 2)} min, batch(" + str(
            1 + int(len_dataset / batch_size)) + "): " + str(i_batch)
        print(text, end="")
        time_batch[1] = time.time()

def export_model(c, path, file_addition=""):
    module_order_dict = {}
    for name, i in c.module_order:
        if i in module_order_dict:
            module_order_dict[i].append(name)
        else:
            module_order_dict[i] = [name]
    depth_max = max(list(module_order_dict.keys()))

    with open(f"{path}/model{file_addition}.txt", "w") as f:
        for i in range(depth_max + 1):
            f.write(f"Depth of Module {i}: \n")
            for key_module in module_order_dict[i]:
                module = c.module[key_module]
                f.write(f"\t Module Name: {key_module} \n")

                f.write(f"\t \t output of the module: \n")
                for neuron_output in module["output"]:
                    f.write(f"\t \t \t Neuron: {neuron_output.name}, Input: {neuron_output.input_keys} \n")
                    f.write(f"\t \t \t \t")
                    for w in list(neuron_output.weights_bias.detach().numpy()):
                        f.write(f"{w},")
                    f.write(f"\n")

                for layer, key_neurons in module["hidden_layer"].items():
                    f.write(f"\t \t Layer within the module {layer}: \n")
                    for key_neuron in key_neurons:
                        neuron = c.model[key_neuron]
                        f.write(f"\t \t \t Neuron: {key_neuron}, Input: {neuron.input_keys} \n")

                        f.write(f"\t \t \t \t")
                        for w in list(neuron.weights_bias.detach().numpy()):
                            f.write(f"{w},")
                        f.write(f"\n")

def create_confusion_matrix(y_true, y_pred, label):
    y_true = y_true.detach().numpy()
    y_true = [label[np.where(l == np.max(l))[0][0]] for l in y_true]

    y_pred = y_pred.detach().numpy()
    y_pred = [label[np.where(l == np.max(l))[0][0]] for l in y_pred]

    return confusion_matrix(y_true, y_pred, labels=label)

def create_confusion_matrix_plt(array, target_names, path):
    """array = np.array([[5607, 1007, 828, 0],
                      [1, 5855, 1586, 0],
                      [198, 1628, 5616, 0],
                      [863, 1996, 4583, 0]])
    """
    array = array

    vmin = np.min(array)
    vmax = np.max(array)

    off_diag_mask = np.eye(*array.shape, dtype=bool)

    #names = ['ctl_CD14p', 'ctl_CD19p', 'ctl_CD3p_CD4p_CD8m', 'ctl_CD3p_CD4m_CD8p']
    array = pd.DataFrame(array, target_names, target_names)

    fig = plt.figure(figsize=(12, 8))  # https://stackoverflow.com/questions/64800003/seaborn-confusion-matrix-heatmap-2-color-schemes-correct-diagonal-vs-wrong-re
    gs0 = matplotlib.gridspec.GridSpec(1, 2, width_ratios=[20, 2], hspace=0.05)
    gs00 = matplotlib.gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs0[1], hspace=0)

    ax = fig.add_subplot(gs0[0])
    cax1 = fig.add_subplot(gs00[0])
    cax2 = fig.add_subplot(gs00[1])

    # https://stackoverflow.com/questions/33104322/auto-adjust-font-size-in-seaborn-heatmap
    sn.heatmap(array, annot=True, annot_kws={"size": 16}, mask=~off_diag_mask, cmap='Greens', vmin=vmin, vmax=vmax, ax=ax, cbar_ax=cax2, fmt='g')  # fmt https://stackoverflow.com/questions/29647749/seaborn-showing-scientific-notation-in-heatmap-for-3-digit-numbers
    sn.heatmap(array, annot=True, annot_kws={"size": 16}, mask=off_diag_mask, cmap='OrRd', vmin=vmin, vmax=vmax, ax=ax, cbar_ax=cax1, cbar_kws=dict(ticks=[]), fmt='g', )

    ax.xaxis.set_ticks_position('top')  # https://stackoverflow.com/questions/49420563/how-can-i-move-the-xlabel-to-the-top
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)  # https://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn-factorplot
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)  # https://stackoverflow.com/questions/26540035/rotate-label-text-in-seaborn-factorplot
    ax.tick_params(axis='both', which='major', labelsize=16)  # https://stackoverflow.com/questions/6390393/matplotlib-make-tick-labels-font-size-smaller
    plt.subplots_adjust(top=0.75)  # https://stackoverflow.com/questions/48526788/python-seaborn-legends-cut-off
    plt.subplots_adjust(left=0.23)

    #sn.set(font_scale=5)
    #plt.show()
    plt.savefig(f"{path}.png")
    plt.close()

def create_ROC(y_true, y_pred, label, path): # https://stackoverflow.com/questions/25009284/how-to-plot-roc-curve-in-python
    for key_dataset in ["test", "val", "train"]:
        y_true_tmp = y_true[key_dataset].detach().numpy()
        y_true2 = y_true_tmp[:, 1]

        y_pred_tmp = y_pred[key_dataset].detach().numpy()
        y_pred2 = y_pred_tmp[:, 1]

        fpr, tpr, _ = metrics.roc_curve(y_true2, y_pred2)
        auc = metrics.roc_auc_score(y_true_tmp, y_pred_tmp)
        plt.plot(fpr, tpr, label=f"{key_dataset}, auc=" + str(auc))
    plt.title(f"{label[0]} vs {label[1]}")
    plt.legend(loc=4)
    #plt.show()
    plt.savefig(f"{path}.png")
    plt.close()

def export_pred(y_true, y_pred, path): #TODO: Wenn Export von konkreten Datensätzen fertig auch Indize von vorhergesamtem Datenpunkt exportieren.
    for dataset_tmp in ["train", "val", "test"]:
        y_true_tmp = y_true[dataset_tmp].detach().numpy()
        y_pred_tmp = y_pred[dataset_tmp].detach().numpy()
        with open(f"{path}/predictions_{dataset_tmp}.txt", "w") as f:
            for i in range(len(y_true_tmp)):
                f.write(str(y_true_tmp[i]))
                f.write(",")
                f.write(str(y_pred_tmp[i]))
                f.write("\n")

def import_pw_names(path):
    delim = ","
    names = {}
    with open(file=path) as f:
        for row in f:
            row = row.rsplit("\t")
            for e in range(3):
                name = row[e].split(delim)
                names[name[1]] = name[2]

            try:
                name_ilmn = row[4].split(delim)[1]
            except IndexError:  # TODO: schnelle Lösung für at
                name_ilmn = row[3].split(delim)[1]
            name_kegg = row[3].split(delim)[2]

            names[name_ilmn] = name_kegg
    return names

def import_ilmn_hsa(path):
    delim = ","
    names = {}
    with open(file=path) as f:
        for row in f:
            row = row.rsplit("\t")
            """for e in range(3):
                name = row[e].split(delim)
                names[name[1]] = name[2]"""
            try:
                name_ilmn = row[4].split(delim)[1]
            except IndexError:  # TODO: schnelle Lösung für at
                name_ilmn = row[3].split(delim)[1]
            name_kegg = row[3].split(delim)[1]

            names[name_ilmn] = name_kegg
    return names

def import_limma(path):
    names = {}
    with open(file=path) as f:
        next(f)
        for row in f:
            row = row.rsplit(",")
            name = row[0].replace('"',"")#.replace("_","")
            names[name] = float(row[5])
    return names