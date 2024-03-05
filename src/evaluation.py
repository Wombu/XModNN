import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sn
import pandas as pd
from sklearn.metrics import confusion_matrix

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


def summerize_conMat(args):
    datasets = ["train", "val", "test"]
    conMat_dict = {d: [] for d in datasets}  # "train": [], "val": [], "test": []
    # metrics_dataset = {"train": copy.deepcopy(metrics_dict), "val": copy.deepcopy(metrics_dict), "test": copy.deepcopy(metrics_dict)}
    for model in args["models"]:
        for dataset_tmp in datasets:
            path = f"{args['path']}/{args['name_model']}/models/{model}/values/confusionMatrix_{dataset_tmp}.txt"
            conMat_dict[dataset_tmp].append(util.import_conMat(path=path))
            #metrics_dict[metric_tmp][dataset_tmp].append(util.import_metric_values(path=path))

    for dataset_tmp in datasets:
        conMat_global = None
        for i in range(len(args["models"])):
            if conMat_global == None:
                conMat_global = conMat_dict[dataset_tmp][i]
            else:
                for r, row in enumerate(conMat_dict[dataset_tmp][i]):
                    for e, ele in enumerate(row):
                        conMat_global[r][e] += ele
        create_confusion_matrix_plt(array=np.array(conMat_global), target_names=args["label"], path=f"{args['path_output_conMat']}/confusionMatrix_{dataset_tmp}")

def create_confusion_matrix(y_true, y_pred, label):
    #y_true = y_true.detach().numpy()
    y_true = [label[np.where(l == np.max(l))[0][0]] for l in y_true]

    y_pred = y_pred.detach().numpy()
    y_pred = [label[np.where(l == np.max(l))[0][0]] for l in y_pred]

    return confusion_matrix(y_true, y_pred, labels=label)