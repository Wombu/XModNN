import copy

from src import Model, util, Controller, Dataset, LRP
from src import component_penalty, component_best_model, component_multiloss, component_metrics
import torch
from shutil import copyfile
import matplotlib.pyplot as plt
import time

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

# output path
path = "output/logic"
util.create_directory(directory_new=path)
path_models = f"{path}/models"
util.create_directory(directory_new=path_models)

source_structure = "data/logic/structure.csv"
source_data = "data/logic/dataset.csv"
source_label = "data/logic/label.csv"

notes = "notes for this model"

# Alle Parameter

#General parameter for the model
args_controler = {"epochs": 12,
                  "batch_size": 8,
                  "disable_bias": False,  # all biases
                  "disable_first_layer_bias": True,  # module specific bias
                  "disable_last_layer_bias": False,  # module specific bias
                  "disable_output_weights_bias": True,  # module specific bias
                  }

# Crossvalidation
args_cake_cv = {"split": 10,  # The number of parts the dataset is divided into.
                "size_cv": 10,  # How many iterations of cross-validation (CV) are performed, typically >= splits
                "size_val": 2,  # How many parts are used for validation
                "size_test": 1}  # How many parts are used for testing

# learning rate
args_optimiser = {"lr": 0.01}

# parameter for the penalty methods
args_penalty = {"method": "L1",  # penalty methods, possibilities: L1, L2, L2_neuron
                "all_weights": True,  # penalty for all weights
                "multiplikator_all_weights": 0.01,
                "last_layer_weights": False,  # module specific penalty
                "multiplikator_last_layer_weights": 0.01,
                "first_layer_weights": True,  # module specific penalty
                "multiplikator_first_layer_weights": 0.1}

# parameter for weight initiation
args_weight_init = {"method": "normal_Xavier",  #initialization, pissibilities: "He", "normal_dist", "Xavier", "normal_Xavier"
                    "mean": 0.0,
                    "std": 0.5}

# parameter for loss functions
args_loss = {"loss": "CrossEntropyLoss",  # loss functions, possibilities: MSELoss, BCELoss
             "disable_weights": True,  # label balancing
             "weights": "automatic"} # automatic or as list for example: [1, 7]

# preprocessing
args_dataPreprocessing = {"mean_reduction": True}

# weighted multi-loss progressive training
args_multiloss = {"threshold_epoch": [2, 4, 6, 6],  # progressive learning
                  "multiloss_weights": {3: 1.3, 2: 1.2, 1: 1.1, 0: 1}}  # weighted layer

# module definition
args_model = {"act": "tanh",  # activation functions, possibilities: "tanh", "sigmoid", "relu"
              "hidden": [3, 3, 3]}  # module size

# calculated metrics
args_metrics = {"metrics": ["acc", "f1", "sens", "spec", "mcc"]}

args_best_model = {"path": path}

args_local_grad = {"path": path}


# Speichern der Parameter, Notizen und Quellorte
sources = {"structure": source_structure, "data": source_data, "label": source_label}

util.notes_to_file(path=path, notes=notes, dicts=[sources, args_controler, args_cake_cv, args_optimiser, args_penalty, args_weight_init, args_loss, args_dataPreprocessing, args_multiloss, args_model])

# Copying the dataset, label, and structure file (omit if the dataset is too large)
copyfile(source_structure, path + "/structure.csv")
util.create_directory(directory_new=f"{path}/data")
copyfile(source_data, f"{path}/data/dataset.csv")
copyfile(source_label, f"{path}/data/label.csv")


features, neurons, output = util.import_structure(filename=source_structure)
data = util.import_data(filename=source_data)
label = util.import_data(filename=source_label)
label_unique_sorted = sorted(list(set(label.iloc[0].tolist())))

data_split = util.data_split_index(data=data, label=label, split=args_cake_cv["split"], size_cv=args_cake_cv["size_cv"], size_val=args_cake_cv["size_val"], size_test=args_cake_cv["size_test"])
util.export_split(path=f"{path}/data", split=data_split)

# Create XMondNN Controler
c = Controller.Controller(args_controler=args_controler, args_multiloss=args_multiloss, args_model=args_model)
c.set_args(args=args_controler)
c.build_model_modules(modules=neurons, features=features, labels=label_unique_sorted, output=output, args_weight_init=args_weight_init)
c.local_graph(modules=neurons)

# Init Components
component_penalty = component_penalty.Penalty(args=args_penalty, method=args_penalty["method"])
c.set_component(component=component_penalty)

component_best_model = component_best_model.Best_model(args=args_best_model)
c.set_component(component=component_best_model)

component_multiloss = component_multiloss.Multiloss(args=args_multiloss)
c.set_component(component=component_multiloss)

component_metric = component_metrics.Metrics(args=args_metrics)
c.set_component(component=component_metric)

for i in range(len(data_split)):
    path_model_iter = f"{path_models}/model_{i}"
    util.create_directory(directory_new=path_model_iter)
    util.create_directory(directory_new=f"{path_model_iter}/values")

    c.components["best_model"].reset_for_iter(path=path_model_iter)

    c.datasets_init(data=data, label=label, data_split=data_split[i], mean_reduction=args_dataPreprocessing["mean_reduction"])

    c.init_weights(args_weight_init=args_weight_init)
    c.disable_bias_func()
    c.set_optimiser(lr=args_optimiser["lr"])
    c.set_loss(args=args_loss)

    time1 = time.time()
    result = c.train_multiloss()  # return: running_loss, running_acc, running_f1, running_sens, running_spec, y_pred_softmax, y_true}
    time2 = time.time()
    print(f"Zeit fürs gesamte Training: {time2 - time1}")

    result_keys = ["acc", "f1", "loss", "sens", "spec", "mcc"]
    #for key, item in result.items():
    for key in result_keys:
        item = result[key]
        plt.title(key)
        plt.figure(figsize=(20, 10))
        plt.plot(item["train"], color="k", label="train")
        plt.plot(item["val"], color="g", label="validation")
        plt.plot(item["test"], color="b", label="test")
        plt.legend()
        plt.savefig(fname=path_model_iter + "/" + key)
        plt.close()

        for key_dataset, item_dataset in item.items():
            with open(f"{path_model_iter}/values/{key}_{key_dataset}.txt", "w") as file:
                for value in item_dataset:
                    file.write(str(value))
                    file.write("\n")

    # global loss
    plt.title("loss_global")
    plt.figure(figsize=(20, 10))
    plt.plot(result["loss_global"], color="k", label="train")
    plt.legend()
    plt.savefig(fname=path_model_iter + "/" + "loss_global")
    plt.close()

    with open(f"{path_model_iter}/values/loss_global_train.txt", "w") as file:
        for value in result["loss_global"]:
            file.write(str(value))
            file.write("\n")

    # confusion-matrix
    for key_dataset in ["test", "val", "train"]:
        confusion_matrix = util.create_confusion_matrix(y_true=result["y_pred"][key_dataset], y_pred=result["y_true"][key_dataset], label=c.label_list)

        with open(f"{path_model_iter}/values/confusionMatrix_{key_dataset}.txt", "w") as file:
            file.write(str(c.label_list))
            file.write("\n")
            for value in confusion_matrix:
                file.write(str(value))
                file.write("\n")

        util.create_confusion_matrix_plt(array=confusion_matrix, target_names=c.label_list, path=f"{path_model_iter}/confusion_matrix_{key_dataset}")

    # Eval und Plot von LRP
    util.create_directory(directory_new=f"{path_model_iter}/LRP")

    #  Epsilon
    lrp = LRP.LRP(controler=c)
    args_lrp = {"epsilon": 0.1, }
    lrp.set_args(args=args_lrp)

    (importances, columnames), (df_importances, df_predictions) = lrp.eval_dataset(dataset="test", type="epsilon", lrp_type="lrp")
    # df_importances = df_importances.set_index(list(importances_dict.keys()))
    # util.plot_importance(importance=importance_modules, module_order=c.module_order, filename=f"{path}/LRP/epsilon_test")
    util.file_importance(importance=importances, columnames=columnames, module_order=c.module_order, modules=c.module, features=c.features, path=f"{path_model_iter}/LRP/epsilon_test_values")
    df_importances.to_csv(f"{path_model_iter}/LRP/epsilon_test_values/importance_dataset_test.csv")
    df_predictions.to_csv(f"{path_model_iter}/LRP/epsilon_test_values/predictions_dataset_test.csv")
    print("Epsilon Test LRP done")

    lrp = LRP.LRP(controler=c)
    args_lrp = {"epsilon": 0.1, }
    lrp.set_args(args=args_lrp)
    (importances, columnames), (df_importances, df_predictions) = lrp.eval_dataset(dataset="train", type="epsilon", lrp_type="lrp")
    # util.plot_importance(importance=importance_nodes, filename=f"{path}/LRP/epsilon_train")
    util.file_importance(importance=importances, columnames=columnames, module_order=c.module_order, modules=c.module, features=c.features, path=f"{path_model_iter}/LRP/epsilon_train_values")
    df_importances.to_csv(f"{path_model_iter}/LRP/epsilon_train_values/importance_dataset_train.csv")
    df_predictions.to_csv(f"{path_model_iter}/LRP/epsilon_train_values/predictions_dataset_train.csv")
    print("Epsilon Train LRP done")

    lrp = LRP.LRP(controler=c)
    args_lrp = {"epsilon": 0.1, }
    lrp.set_args(args=args_lrp)
    (importances, columnames), (df_importances, df_predictions) = lrp.eval_dataset(dataset="validation", type="epsilon", lrp_type="lrp")
    # util.plot_importance(importance=importance_nodes, filename=f"{path}/LRP/epsilon_val")
    util.file_importance(importance=importances, columnames=columnames, module_order=c.module_order, modules=c.module, features=c.features, path=f"{path_model_iter}/LRP/epsilon_val_values")
    df_importances.to_csv(f"{path_model_iter}/LRP/epsilon_val_values/importance_dataset_validation.csv")
    df_predictions.to_csv(f"{path_model_iter}/LRP/epsilon_val_values/predictions_dataset_validation.csv")
    print("Epsilon Val LRP done")
