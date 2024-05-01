import pandas as pd
import numpy as np
import copy
from src import util_cv
import matplotlib.pyplot as plt
from sklearn import metrics
import matplotlib
import seaborn as sn

def summerize_lrp(args, lrp_type_dataset):
    pathway_important = {}

    # Iterieren über die Tiefe der Hierarchie
    for d in range(1, args["depth"]):
        # Output der wichtigen Pathways
        pathway_important[d] = []

        # Erstellung der directorys der Teife
        dir_depth = f"{args['lrp']}/depth_{d}"
        util_cv.create_directory(directory_new=dir_depth)

        # import lrp-Werte aller model einer Tiefe
        dict_import = {}
        for model in args["models"]:
            # Import
            path = f"{args['path_model']}/models/{model}/LRP/{lrp_type_dataset}/depth_{d}.csv"
            data_new, header = util_cv.import_lrp(path=path)

            # Erweiterung von dict_import und anfügen der importierten Werte
            if not dict_import:
                dict_import = {key_metric: {key_module: [] for key_module in data_new.keys()} for key_metric in header[1:]}

            for m, key_metric in enumerate(header[1:]):
                for key_module in data_new.keys():
                    dict_import[key_metric][key_module].append(data_new[key_module][m])
                    dict_import[key_metric][f"{key_module}_sum"] = np.sum(dict_import[key_metric][key_module]) # Aussortieren von nicht bewerteten pathways

        # Lösung von leeren Vorhersagen (certainty meist c >= 0.7)
        key_metric = list(dict_import.keys())
        key_modules = list(data_new.keys())
        for key_metric in key_metric:
            for key_module in key_modules:
                if dict_import[key_metric][f"{key_module}_sum"] == 0:
                    del dict_import[key_metric][key_module]
                del dict_import[key_metric][f"{key_module}_sum"]

            if len(dict_import[key_metric]) == 0:
                del dict_import[key_metric]

        # Umformung in dataframes
        for key_metric in dict_import.keys():
            dict_import[key_metric] = pd.DataFrame.from_dict(dict_import[key_metric], orient="index")

        results = {}
        # Über alle Metriken und einer Tiefe (durch d) iterieren
        for key_metric in dict_import.keys():
            results[key_metric] = {}
            relevances = dict_import[key_metric]


            filter = relevances.index.str.contains("_residual") #residuals raussortieren
            relevances = relevances[~filter]

            relevances_ratio = relevances.div(relevances.sum(axis=0))

            # Erstellung der [f"mean_{method}", f"median_{method}", f"std_{method}",  f"min_{method}",  f"max_{method}"] - Spalten
            results[key_metric] = util_cv.combine_results(relevances_ratio, method="ratio")

            # Anfügen der Namen der Pathways und Proteinen (ILMN_ID)
            results[key_metric]["names"] = results[key_metric].index.to_series().map(args["rename"])

            # Anfügen der Limma-Bewertungen für Layer D
            if d == args["depth"]-1:
                results[key_metric]["hsa"] = results[key_metric].index.to_series().map(args["ilmn_hsa"])

            # Erstellung der Thresholds, ab wann eine Bewertung relevant ist.
            std_threshold = np.std(results[key_metric]["mean_ratio"])
            results[key_metric][f"over_std_mean_ratio"] = False
            results[key_metric][f"over_std_mean_ratio"][results[key_metric]["mean_ratio"] > std_threshold] = True

            std_threshold = np.std(results[key_metric]["median_ratio"]) + np.mean(results[key_metric]["median_ratio"])
            results[key_metric][f"over_std_median_ratio"] = False
            results[key_metric][f"over_std_median_ratio"][results[key_metric]["median_ratio"] > std_threshold] = True


            results[key_metric].to_csv(f"{dir_depth}/{key_metric}.csv")
            if d == 4:
                results[key_metric][["names", "median_ratio"]][results[key_metric]["over_std_median_ratio"]].to_csv(f"{dir_depth}/{key_metric}_cytoscape.csv", sep=",", index=False)  # Für Stefan besonderer seperator und nur einzelne Spalten

            results[key_metric].index.name = 'names_neuron'
            results[key_metric].reset_index(inplace=True)

        # Sammeln der wichtigen Pathways, um später alle dazugehörigen Elemente zu vereinen und zu exportieren.
        pathway_important[d] = copy.copy(results["0.5_sum"])

    return pathway_important

def summerize_best_models(args):
    metrics = ["acc", "f1", "loss", "sens", "spec", "mcc", "ep"]
    metrics_dict = {m: [] for m in metrics}
    metrics_dataset = {"train": copy.deepcopy(metrics_dict), "val": copy.deepcopy(metrics_dict), "test": copy.deepcopy(metrics_dict)}
    for model in args["models"]:
        path = f"{args['path_model']}/models/{model}/best_model/metric_values.csv"
        data_new, header = util_cv.import_best_model_values(path=path)
        for metric_name_tmp in data_new.keys():
            for d, data_name_tmp in enumerate(header[1:]):
                metrics_dataset[data_name_tmp][metric_name_tmp].append(data_new[metric_name_tmp][d])


    colnames = ["mean", "median", "std", "min", "max"]
    summerised_dataset = {"train": copy.deepcopy(metrics_dict), "val": copy.deepcopy(metrics_dict), "test": copy.deepcopy(metrics_dict)}
    for data_name_tmp in metrics_dataset.keys():
        for metric_name_tmp in metrics_dataset[data_name_tmp].keys():
            summerised_dataset[data_name_tmp][metric_name_tmp].append(np.mean(metrics_dataset[data_name_tmp][metric_name_tmp]))
            summerised_dataset[data_name_tmp][metric_name_tmp].append(np.median(metrics_dataset[data_name_tmp][metric_name_tmp]))
            summerised_dataset[data_name_tmp][metric_name_tmp].append(np.std(metrics_dataset[data_name_tmp][metric_name_tmp]))
            summerised_dataset[data_name_tmp][metric_name_tmp].append(np.min(metrics_dataset[data_name_tmp][metric_name_tmp]))
            summerised_dataset[data_name_tmp][metric_name_tmp].append(np.max(metrics_dataset[data_name_tmp][metric_name_tmp]))

    for data_name_tmp in metrics_dataset.keys():
        df = pd.DataFrame.from_dict(summerised_dataset[data_name_tmp], orient="index")
        df.columns = colnames

        df.to_csv(f"{args['best_models']}/best_models_{data_name_tmp}.csv")

def summerize_metrics(args):
    metrics = ["acc", "f1", "loss", "sens", "spec"]
    datasets = ["train", "val", "test"]
    metrics_dict = {m: {d: [] for d in datasets} for m in metrics} # "train": [], "val": [], "test": []
    for model in args["models"]:
        for metric_tmp in metrics_dict.keys():
            for dataset_tmp in metrics_dict[metric_tmp].keys():
                path = f"{args['path_model']}/models/{model}/values/{metric_tmp}_{dataset_tmp}.txt"
                metrics_dict[metric_tmp][dataset_tmp].append(util_cv.import_metric_values(path=path))

    color = {"test": "blue", "train": "green", "val": "orange"}
    title = {"test": "Testset", "train": "Trainingsset", "val": "Validationset"}
    metrics_description = {"acc": "accuracy", "f1": "f1-score", "loss": "loss", "sens":"sensitivity", "spec":"specificity"}

    for metric_tmp in metrics:
        fig, axs = plt.subplots(2, 2, figsize=(20, 10))
        axs = [axs[0][0], axs[0][1], axs[1][0], axs[1][1]]
        y_lim = []
        for d, dataset_tmp in enumerate(datasets):
            data = metrics_dict[metric_tmp][dataset_tmp]
            df = pd.DataFrame(data)
            shape = df.shape
            mean = df.mean()
            min_v = df.min()
            max_v = df.max()
            x = range(0, shape[1])
            axs[d].plot(x, mean, color=color[dataset_tmp],)
            axs[d].fill_between(x, min_v, max_v, alpha=0.2, color=color[dataset_tmp], label="MinMax")
            axs[d].set_title(title[dataset_tmp])

            axs[3].plot(x, mean, color=color[dataset_tmp])
            y_lim.append(axs[d].get_ylim())

        axs[3].set_title("Combined")

        y_lim_0 = min([p[0] for p in y_lim])
        y_lim_1 = max([p[1] for p in y_lim])
        y_lim = tuple([y_lim_0, y_lim_1])
        plt.setp(axs, ylim=y_lim)

        caption = f"Displays the {metrics_description[metric_tmp]} throughout the training epochs"
        plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)

        plt.savefig(f"{args['path_output_metric']}/{metric_tmp}.png", transparent=True)
        plt.close()

def summerize_metrics_special(args):
    metrics = ["acc", "f1", "loss", "sens", "spec"]
    datasets = ["train", "val", "test"]
    metrics_dict = {m: {d: [] for d in datasets} for m in metrics} # "train": [], "val": [], "test": []
    #metrics_dataset = {"train": copy.deepcopy(metrics_dict), "val": copy.deepcopy(metrics_dict), "test": copy.deepcopy(metrics_dict)}
    for model in args["models"]:
        for metric_tmp in metrics_dict.keys():
            for dataset_tmp in metrics_dict[metric_tmp].keys():
                path = f"{args['path_model']}/models/{model}/values/{metric_tmp}_{dataset_tmp}.txt"
                metrics_dict[metric_tmp][dataset_tmp].append(util_cv.import_metric_values(path=path))

    color = {"test": "blue", "train": "green", "val": "orange"}
    title = {"test": "Testset", "train": "Trainingsset", "val": "Validationset"}
    metrics_description = {"acc": "accuracy", "f1": "f1-score", "loss": "loss", "sens":"sensitivity", "spec":"specificity"}

    for metric_tmp in metrics:
        fig, axs = plt.subplots(1, 2, figsize=(20, 5))
        #plt.tight_layout()
        #axs = [axs[0][0], axs[0][1]]
        y_lim = []
        for d, dataset_tmp in enumerate(["train", "val"]):
            data = metrics_dict[metric_tmp][dataset_tmp]
            df = pd.DataFrame(data)
            shape = df.shape
            mean = df.mean()
            min_v = df.min()
            max_v = df.max()
            x = range(0, shape[1])
            axs[0].plot(x, mean, color=color[dataset_tmp], label=f"mean {metric_tmp}, {title[dataset_tmp]}")
            axs[0].fill_between(x, min_v, max_v, alpha=0.2, color=color[dataset_tmp], label="MinMax")

            #axs[3].plot(x, mean, color=color[dataset_tmp])
            y_lim.append(axs[0].get_ylim())
        axs[0].set_title("Training and Validation")
        axs[0].legend(loc='center right')


        data = metrics_dict[metric_tmp]["test"]
        df = pd.DataFrame(data)
        shape = df.shape
        mean = df.mean()
        min_v = df.min()
        max_v = df.max()
        x = range(0, shape[1])
        axs[1].plot(x, mean, color=color["test"], label=f"mean {metric_tmp}, {title['test']}")
        axs[1].fill_between(x, min_v, max_v, alpha=0.2, color=color["test"], label="MinMax")
        axs[1].set_title(title["test"])
        axs[1].legend(loc='center right')

        #axs[3].plot(x, mean, color=color[dataset_tmp])
        y_lim.append(axs[1].get_ylim())

        y_lim_0 = min([p[0] for p in y_lim])
        y_lim_1 = max([p[1] for p in y_lim])
        y_lim = tuple([y_lim_0, y_lim_1])
        plt.setp(axs, ylim=y_lim)

        caption = f"Displays the {metrics_description[metric_tmp]} throughout the training epochs"
        plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)
        #plt.axes().set_aspect('equal')

        #plt.show()
        plt.savefig(f"{args['path_output_metric']}/{metric_tmp}_special.png", transparent=True)
        plt.close()

def summerize_metrics_single(args):
    metrics = ["acc", "f1", "loss", "sens", "spec"]
    datasets = ["train", "val", "test"]

    color = {"test": "blue", "train": "green", "val": "orange"}
    title = {"test": "Testset", "train": "Trainingsset", "val": "Validationset"}
    metrics_description = {"acc": "accuracy", "f1": "f1-score", "loss": "loss", "sens": "sensitivity", "spec": "specificity"}

    for model in args["models"]:

        metrics_dict = {m: {d: None for d in datasets} for m in metrics}  # "train": [], "val": [], "test": []
        #metrics_dataset = {"train": copy.deepcopy(metrics_dict), "val": copy.deepcopy(metrics_dict), "test": copy.deepcopy(metrics_dict)}

        for metric_tmp in metrics_dict.keys():
            for dataset_tmp in metrics_dict[metric_tmp].keys():
                path = f"{args['path_model']}/models/{model}/values/{metric_tmp}_{dataset_tmp}.txt" #TODO: hier richtige Werte importieren
                metrics_dict[metric_tmp][dataset_tmp] = util_cv.import_metric_values(path=path)

        for metric_tmp in metrics:
            fig, axs = plt.subplots(1, 1, figsize=(9, 5))
            #plt.tight_layout()
            #axs = [axs[0][0], axs[0][1]]
            y_lim = []
            for dataset_tmp in datasets:
                data = metrics_dict[metric_tmp][dataset_tmp]
                x = list(range(len(data)))
                axs.plot(x, data, color=color[dataset_tmp], label=f"{metric_tmp}, {title[dataset_tmp]}")

                y_lim.append(axs.get_ylim())
            axs.set_title(f"{metric_tmp} for Training-, Validation- and Testset")
            axs.legend(loc='center right')

            caption = f"Displays the {metrics_description[metric_tmp]} throughout the training epochs"
            plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)
            #plt.axes().set_aspect('equal')

            #plt.show()
            plt.savefig(f"{args['path_output_metric']}/{model}_{metric_tmp}.png", transparent=True)
            plt.close()

def ROC_global(args):
    classes_num = len(args["label"])
    datasets = ["train", "val", "test"]
    roc_dict = {d: [] for d in datasets}  # "train": [], "val": [], "test": []
    # metrics_dataset = {"train": copy.deepcopy(metrics_dict), "val": copy.deepcopy(metrics_dict), "test": copy.deepcopy(metrics_dict)}
    for model in args["models"]:
        for dataset_tmp in datasets:
            path = f"{args['path_model']}/models/{model}/values/predictions_{dataset_tmp}.txt"
            roc_dict[dataset_tmp].append(util_cv.import_pred(path=path))

    color = {"test": "blue", "train": "green", "val": "orange"}
    title = {"test": "Testset", "train": "Trainingsset", "val": "Validationset"}

    fig, axs = plt.subplots(2, 2, figsize=(20, 10))
    axs = [axs[0][0], axs[0][1], axs[1][0], axs[1][1]]

    #for dataset_tmp in datasets:
    for d, dataset_tmp in enumerate(datasets):
        roc_global = {"auc": [], "tpr": [], "fpr":[]}
        for i_model in range(len(args["models"])):
            y_true_model = np.array(roc_dict[dataset_tmp][i_model][0])
            y_pred_model = np.array(roc_dict[dataset_tmp][i_model][1])
            # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
            fpr = {}
            tpr = {}
            fpr_grid = np.linspace(0.0, 1.0, 1000)
            for class_num in range(classes_num):
                y_true_class = y_true_model[:, class_num]
                y_pred_class = y_pred_model[:, class_num]

                fpr[class_num], tpr[class_num], _ = metrics.roc_curve(y_true_class, y_pred_class)

            tpr_mean = np.zeros_like(fpr_grid)
            for i in range(classes_num):
                tpr_mean += np.interp(fpr_grid, fpr[i], tpr[i])

            tpr_mean /= classes_num

            roc_global["tpr"].append(tpr_mean)
            auc = metrics.auc(fpr_grid, tpr_mean)
            roc_global["auc"].append(auc)

        df = pd.DataFrame(roc_global["tpr"])
        mean = df.mean()
        min = df.min()
        max = df.max()
        axs[d].plot(fpr_grid, mean, color=color[dataset_tmp], label=f"mean, auc=" + str(round(np.mean(roc_global['auc']), 2)))
        axs[d].fill_between(fpr_grid, min, max, alpha=0.2, color=color[dataset_tmp], label=f"MinMax, auc=[{str(round(np.min(roc_global['auc']), 2))};{str(round(np.max(roc_global['auc']), 2))}]")
        axs[d].set_title(f"ROC-Curves, {len(args['models'])} fold-crossvalidation: {title[dataset_tmp]}")
        axs[d].legend(loc=4)
        axs[3].plot(fpr_grid, mean, color=color[dataset_tmp], label=f"{dataset_tmp}, auc=" + str(round(np.mean(roc_global['auc']),2)))


    axs[3].set_title("Combined")
    axs[3].legend(loc=4)

    caption = f"Displays the ROC-Curves and their minimum and maximum of the {len(args['models'])} fold-crossvalidation"
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)
    plt.setp(axs, ylim=(0, 1.01), xlim=(0, 1))

    plt.savefig(f"{args['path_output_roc']}/global_ROC.png", transparent=True)
    #plt.show()
    plt.close()

def ROC_single(args):
    classes_num = len(args["label"])
    datasets = ["train", "val", "test"]
    roc_dict = {d: [] for d in datasets}  # "train": [], "val": [], "test": []
    for model in args["models"]:
        for dataset_tmp in datasets:
            path = f"{args['path_model']}/models/{model}/values/predictions_{dataset_tmp}.txt"
            roc_dict[dataset_tmp].append(util_cv.import_pred(path=path))

    color = {"test": "blue", "train": "green", "val": "orange"}

    for i_model in range(len(args["models"])):
        fig, axs = plt.subplots(1, 1, figsize=(10, 5))
        for d, dataset_tmp in enumerate(datasets):
            y_true_model = np.array(roc_dict[dataset_tmp][i_model][0])
            y_pred_model = np.array(roc_dict[dataset_tmp][i_model][1])
            # https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
            fpr = {}
            tpr = {}
            fpr_grid = np.linspace(0.0, 1.0, 1000)
            for class_num in range(classes_num):
                y_true_class = y_true_model[:, class_num]
                y_pred_class = y_pred_model[:, class_num]

                fpr[class_num], tpr[class_num], _ = metrics.roc_curve(y_true_class, y_pred_class)

            tpr_mean = np.zeros_like(fpr_grid)
            for i in range(classes_num):
                tpr_mean += np.interp(fpr_grid, fpr[i], tpr[i])

            tpr_mean /= classes_num
            auc = metrics.auc(fpr_grid, tpr_mean)

            axs.plot(fpr_grid, tpr_mean, color=color[dataset_tmp], label=f"{dataset_tmp}, auc=" + str(round(np.mean(auc), 2)))
            axs.set_title(f"ROC Curve, best model")
        axs.legend(loc=4)

        plt.setp(axs, ylim=(0, 1.01), xlim=(0, 1))

        plt.savefig(f"{args['path_output_roc']}/ROC_model_{i_model}.png", transparent=True)

        plt.close()

def summerize_conMat(args):
    datasets = ["train", "val", "test"]
    conMat_dict = {d: [] for d in datasets}  # "train": [], "val": [], "test": []
    for model in args["models"]:
        for dataset_tmp in datasets:
            path = f"{args['path_model']}/models/{model}/values/confusionMatrix_{dataset_tmp}.txt"
            conMat_dict[dataset_tmp].append(util_cv.import_conMat(path=path))

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

def create_confusion_matrix_plt(array, target_names, path):
    vmin = np.min(array)
    vmax = np.max(array)

    off_diag_mask = np.eye(*array.shape, dtype=bool)

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

    plt.savefig(f"{path}.png", transparent=True)
    plt.close()

def summerize_single_predictios(args, lrp_type_dataset, important_pw, path_boxplot):
    # Iterieren über die Tiefe der Hierarchie
    for d in range(1, args["depth"]):

        # import lrp-Werte aller model einer Tiefe
        dict_import = {}
        for model in args["models"]:
            # Import
            path = f"{args['path_model']}/models/{model}/LRP/{lrp_type_dataset}/depth_{d}.csv"
            data_new, header = util_cv.import_lrp(path=path)

            keys_tmp = list(data_new.keys())

            for key_module in keys_tmp:
                if not (key_module.find("residual") == -1):
                    del data_new[key_module]

            # Erweiterung von dict_import und anfügen der importierten Werte
            if not dict_import:
                dict_import = {key_metric: {key_module: [] for key_module in data_new.keys()} for key_metric in header[1:]}

            for m, key_metric in enumerate(header[1:]):
                for key_module in data_new.keys():
                    dict_import[key_metric][key_module].append(float(data_new[key_module][m]))


        data_boxplot = dict_import["0.5_sum"]
        sum_dict = {i: 0 for i in range(10)}
        for pw in data_boxplot.keys():
            for i in range(10): #10, wegen 10 fold crossvalidation
                sum_dict[i] += data_boxplot[pw][i]


        min_boxplot = float("inf")
        max_boxplot = -float("inf")
        for pw in important_pw:
            if pw in data_boxplot:
                name = util_cv.name_short(args["rename"][pw])
                data_boxplot[pw] = [item / sum_dict[i] for i, item in enumerate(data_boxplot[pw])]  # ! geht nicht!
                min_tmp = min(data_boxplot[pw])
                max_tmp = max(data_boxplot[pw])

                if min_tmp < min_boxplot:
                    min_boxplot = min_tmp

                if max_tmp > max_boxplot:
                    max_boxplot = max_tmp

                #fig, ax = plt.subplots(figsize=(2, 4))
                #fig = plt.figure(figsize=(2, 4), dpi=100)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4, 4))
                ax1.boxplot(data_boxplot[pw], widths=[0.75])
                ax1.set_ylabel("Normalized lrp importance")
                ax1.tick_params(
                    axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)

                ax2.boxplot(data_boxplot[pw],  widths=[0.75]) #labels=[name]
                ax2.set_ylabel("y-limit min/max of the hole layer")
                y_lim_tol = (max_boxplot - min_boxplot) * 0.05
                ax2.set_ylim([min_boxplot - y_lim_tol, max_boxplot + y_lim_tol])
                ax2.tick_params(
                    axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom=False,  # ticks along the bottom edge are off
                    top=False,  # ticks along the top edge are off
                    labelbottom=False)

                #plt.tight_layout()
                #plt.show()
                fig.text(.5, -.05, name, ha='center')
                fig.tight_layout()
                plt.savefig(f"{path_boxplot}/lvl{d}_{name}.png", bbox_inches="tight")
                plt.close()

def export_per_pw(args, importances, path):
    df_importances = []
    for d in importances.keys():
            importances[d]["depth"] = d
            df_importances.append(importances[d])
    df_importances = pd.concat(df_importances)

    modules_imp = df_importances[df_importances["over_std_median_ratio"]]
    modules_imp_related = get_all_related_pw(module_names=modules_imp["names_neuron"], module_hierarchy=args["module_hierarchy"], features=args["features"])

    for module_name, modules_related in modules_imp_related.items():
        modules_related.add(module_name)
        df_importances[df_importances["names_neuron"].isin(modules_related)].to_csv(f"{path}/depth{args['modules_depth'][module_name]}_{module_name}.csv", sep="\t")

def get_all_related_pw(module_names, module_hierarchy, features):
    modules_related = {}
    for module_imp in module_names:
        if module_imp in features:
            continue

        modules_related[module_imp] = []
        queue = module_hierarchy[module_imp]
        while len(queue) != 0:
            module_pop = queue.pop()
            modules_related[module_imp].append(module_pop)
            if module_pop not in features:
                queue = queue + module_hierarchy[module_pop]
        modules_related[module_imp] = set(modules_related[module_imp])
    return modules_related