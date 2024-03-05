import copy

from src import Model, util, Controller, Dataset, LRP
from src import component_penalty, component_early_stopping, component_best_model, component_multiloss, component_metrics
import torch
from shutil import copyfile
import matplotlib.pyplot as plt
import time

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

path = "output/logic"

util.create_directory(directory_new=path)
path_models = f"{path}/models"
util.create_directory(directory_new=path_models)

notes = "ohne! Residual Transfer und ohne Residual Abzug"

num_model = 10

# Alle Parameter
#device = "cuda:1"
device = "cpu"

#torch.manual_seed(3)

args_controler = {"epochs": 10,
                  "batch_size": 64,
                  "validation_size": 0.2,
                  "test_size": 0.2,
                  "disable_bias": False,
                  "disable_first_layer_bias": True,
                  "disable_last_layer_bias": False,
                  "disable_output_weights_bias": True,
                  "device": torch.device(device if torch.cuda.is_available() else 'cpu')
                  }


args_optimiser = {"lr": 0.01}

args_penalty = {"method": "L1",  # L1, L2, L2_neuron
                "all_weights": True,
                "multiplikator_all_weights": 0.01,
                "last_layer_weights": False,
                "multiplikator_last_layer_weights": 0.01,
                "first_layer_weights": True,
                "multiplikator_first_layer_weights": 0.1}

args_weight_init = {"method": "normal_Xavier", #"He", "normal_dist", "Xavier", "normal_Xavier"
                    "mean": 0.0,
                    "std": 0.5}

args_loss = {"loss": "CrossEntropyLoss", #MSELoss, BCELoss
             "disable_weights": True,
             "weights": [3, 1]}

args_dataPreprocessing = {"mean_reduction": True,
                          "balanced_dataset": True}

"""args_early_stopping = {"method": "stopping_rising_error_lazy",
                       #"error_threshold": 0.4,  # bei lazy nicht benötigt
                       "stop_lazy": 100}"""

args_best_model = {"path": path}

args_local_grad = {"path": path}

args_metrics = {"metrics": ["acc", "f1", "sens", "spec", "mcc"]}

args_multiloss = {"threshold_epoch": [4, 5, 6, 6],
                  "multiloss_weights": {3: 1.3, 2: 1.2, 1: 1.1, 0: 1}}
args_model = {"act": "tanh",  #"tanh", "sigmoid", "relu"
              "hidden": [3, 3, 3]}

source_structure = "data/logic/logic_structure.csv"
source_data = "data/logic/logic_and_or_dataset.csv"
source_label = "data/logic/logic_and_or_label.csv"


# Speichern der Parameter, Notizen und Quellorte
sources = {"structure": source_structure, "data": source_data, "label": source_label}

util.notes_to_file(path=path, notes=notes, dicts=[sources, args_dataPreprocessing,  args_controler, args_optimiser, args_penalty, args_weight_init, args_loss, args_multiloss, args_model]) #args_early_stopping

# Kopieren des Datensatzes, Label und Strukturdatei (Bei zu großen Datensätzen weglassen)
copyfile(source_structure, path + "/structure.txt")
util.create_directory(directory_new=f"{path}/data")
copyfile(source_data, f"{path}/data/dataset.txt")
copyfile(source_label, f"{path}/data/label.txt")
#copyfile("data/artificialDatasets/data/logic.txt", path + "/logic.txt")


features, neurons, output = util.import_structure(filename=source_structure)
data, data_colnames = util.import_data(filename=source_data)
label, label_colnames = util.import_data(filename=source_label)

if(args_dataPreprocessing["mean_reduction"]): #! TODO: export values
    for key, item in data.items():
        data[key] = util.mean_reduction(values=item)

dataset = Dataset.Dataset(x=data, y=label)
if(args_dataPreprocessing["balanced_dataset"]):
    dataset.balance()

# Create mondNN Controler
c = Controller.Controller(args_controler=args_controler, args_multiloss=args_multiloss, args_model=args_model)
c.set_args(args=args_controler)
c.build_model_modules(modules=neurons, features=features, labels=dataset.label_list, output=output, args_weight_init=args_weight_init)
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

for i in range(num_model):
    path_model_iter = f"{path_models}/model_{i}"
    util.create_directory(directory_new=path_model_iter)
    util.create_directory(directory_new=f"{path_model_iter}/values")
    util.create_directory(directory_new=f"{path_model_iter}/module_eval")

    c.components["best_model"].reset_for_iter(path=path_model_iter)

    c.init_weights(args_weight_init=args_weight_init)
    c.disable_bias_func()
    c.set_optimiser(lr=args_optimiser["lr"])
    c.set_loss(args=args_loss)

    c.datasets_init(dataset=copy.deepcopy(dataset)) # TODO: deppkopy frisst viel Zeit.
    #TODO: Speichern von konkreten Datasets, bzw den verwendeten Indizes? Langfristig inkl Methode zum erneuten Einlesen der verwendeten Datensätze.

    time1 = time.time()
    result = c.train_multiloss()  # return: running_loss, running_acc, running_f1, running_sens, running_spec, y_pred_softmax, y_true}
    time2 = time.time()
    print(f"Zeit fürs gesamte Training: {time2 - time1}")

    # Export model structure
    util.export_model_structure(c=c, path=path_model_iter)
    # Error as Plot
    #util.export_module_error(module_error=c.error_module, module_error_ep=c.error_module_ep, path=f"{path_model_iter}/module_eval")

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

    # ROC-Curve, TODO: noch binary! https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
    util.export_pred(y_true=result["y_true"], y_pred=result["y_pred"], path=f"{path_model_iter}/values")
    #util.create_ROC(y_true=result["y_true"], y_pred=result["y_pred"], label=c.label_list, path=f"{path_model_iter}/ROC") # in eval ausgelagert

    # Eval und Plot von LRP
    util.create_directory(directory_new=f"{path_model_iter}/LRP")

    #  Epsilon
    if(args_controler["test_size"] != 0):
        lrp = LRP.LRP(controler=c)
        args_lrp = {"epsilon": 0.1, }
        lrp.set_args(args=args_lrp)

        importancesb, columnames = lrp.eval_dataset(dataset="test", type="epsilon", lrp_type="lrp")
        #util.plot_importance(importance=importance_modules, module_order=c.module_order, filename=f"{path}/LRP/epsilon_test")
        util.file_importance(importance=importancesb, columnames=columnames, module_order=c.module_order, modules=c.module, features=c.features, path=f"{path_model_iter}/LRP/epsilon_test_lrp")
        print("Epsilon Test LRP done")

    lrp = LRP.LRP(controler=c)
    args_lrp = {"epsilon": 0.1, }
    lrp.set_args(args=args_lrp)
    importances, columnames = lrp.eval_dataset(dataset="train", type="epsilon", lrp_type="lrp")
    #util.plot_importance(importance=importance_nodes, filename=f"{path}/LRP/epsilon_train")
    util.file_importance(importance=importances, columnames=columnames, module_order=c.module_order, modules=c.module, features=c.features, path=f"{path_model_iter}/LRP/epsilon_train_values")
    print("Epsilon Train LRP done")

    lrp = LRP.LRP(controler=c)
    args_lrp = {"epsilon": 0.1, }
    lrp.set_args(args=args_lrp)
    importances, columnames = lrp.eval_dataset(dataset="validation", type="epsilon", lrp_type="lrp")
    #util.plot_importance(importance=importance_nodes, filename=f"{path}/LRP/epsilon_val")
    util.file_importance(importance=importances, columnames=columnames, module_order=c.module_order, modules=c.module, features=c.features, path=f"{path_model_iter}/LRP/epsilon_val_values")
    print("Epsilon Val LRP done")

