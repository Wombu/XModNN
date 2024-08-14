from src.Model import Neuron
from src import general_eval as g_eval
from src import util, Dataset

"""from Model import Neuron
import general_eval as g_eval
import util"""

import torch
from torch.utils import data
import torch.multiprocessing as mp

import numpy as np
import sys
import copy
#from src import util_experimental as util_exp
import time
#from numba import njit, prange

from sys import getsizeof

class Controller():
    def __init__(self, args_controler, args_multiloss, args_model):
        self.args_controler = args_controler
        """Dict of arguments for Controller"""
        self.args_multiloss = args_multiloss
        self.args_model = args_model
        self.model = None
        self.output = None
        self.features = None

        self.optimizer = None
        self.var_trainable = None
        self.loss = None

        self.module = {}  # Beinhaltet List: Output-Neuronen, depth, Zusatzneuronen

        # TODO: vielleicht ohne vordefiniert zu sein.
        self.components = {"penalty": None,
                           "early_stopping": None,
                           "best_model": None,
                           "multiloss": None,
                           "local_grad": None,
                           "metric": None}

        self.error_module = {}
        self.error_module_ep = []

        self.training_mode = False
        #self.x_batch = None  # Wird benötigt, damit die Daten nicht andauern kopiert werden. Leider -.-

    #! Für early stopping, könnte es Sinn machen mehrere Kriterien (components) zu nutzten -> mit Array von components arbeiten.
    def set_component(self, component):  #! Mit Typabfrage besser? Anstelle von string.
        if component.name_component in self.components:
            component.c = self
            self.components[component.name_component] = component
            self.components[component.name_component].init()
        else:
            print("Component " + component.name_component + " not implemented")

    def datasets_init(self, data, label, data_split, mean_reduction): #TODO: schöner programmieren
        if mean_reduction:
            data_tmp = data.iloc[:, data_split[0]]
            mean = data_tmp.mean(axis=0)
            data_tmp = data_tmp - mean
            mean = mean.mean()
            label_tmp = label.iloc[:, data_split[0]]
            self.dataset_train = Dataset.Dataset(x=copy.copy(data_tmp), y=label_tmp)

            data_tmp = data.iloc[:, data_split[1]]
            data_tmp = data_tmp - mean.mean()
            label_tmp = label.iloc[:, data_split[1]]
            self.dataset_validation = Dataset.Dataset(x=copy.copy(data_tmp), y=label_tmp)

            data_tmp = data.iloc[:, data_split[2]]
            label_tmp = label.iloc[:, data_split[2]]
            data_tmp = data_tmp - mean
            self.dataset_test = Dataset.Dataset(x=copy.copy(data_tmp), y=label_tmp)
        else:
            data_tmp = data.iloc[:, data_split[0]]
            label_tmp = label.iloc[:, data_split[0]]
            self.dataset_train = Dataset.Dataset(x=data_tmp, y=label_tmp)

            data_tmp = data.iloc[:, data_split[1]]
            label_tmp = label.iloc[:, data_split[1]]
            self.dataset_validation = Dataset.Dataset(x=data_tmp, y=label_tmp)

            data_tmp = data.iloc[:, data_split[2]]
            label_tmp = label.iloc[:, data_split[2]]
            self.dataset_test = Dataset.Dataset(x=data_tmp, y=label_tmp)

    # Erstelle für jeden Pathway ein Modul, welches aus min einer Schicht mit so vielen Neuronen wie es label gibt, besteht
    def build_model_modules(self, modules, features, labels, output, args_weight_init):
        self.output = output
        self.label_list = labels
        self.features = features
        self.model = {}

        # ! TODO: kompakter: function init_module, create_neurons,
        # ! TODO: Modul als vollständiges Netz behandeln? Mit seq etz...
        # for module_name, module_act, module_input in modules:
        for module_name, module_input in modules:
            self.module[module_name] = {"output": [], "input": [], "depth": None, "hidden": self.args_model["hidden"], "neurons": []}  # ! TODO: Module als Klasse?  #! TODO:hidden variable machen, vllt in Abhängigkeit von Größe des Inputs[3, 3]
            self.module[module_name]["hidden_layer"] = {i: [] for i in range(len(self.module[module_name]["hidden"]))}
            # Erstellung der hidden Layer für jedes Modul
            for layer_pos, layer_size in reversed(list(enumerate(self.module[module_name]["hidden"]))):
                # Zunächst die Erstellung der eindeutigen Strings des Inputs der Neuronen
                if layer_pos == len(self.module[module_name]["hidden"]) - 1:  # Input von anderen Modulen, da "len(hidden)" -1 der erste layer ist
                    input_tmp = []
                    for module_input_name in module_input:  # Input des Moduls übernehmen
                        if module_input_name in self.features:  # Wenn Input Feature, nur String übernehmen
                            input_tmp.append(module_input_name)
                        else:  # Wenn der Input von einem anderen Modul kommt, erstelle Strings zu den konkreten Neuronen von letztem Layer des Input modules
                            size_last_layer_input = self.module[module_input_name]['hidden'][0]  # [-1]
                            input_tmp = input_tmp + [f"{module_input_name}_0_{n_list_comp}" for n_list_comp in range(size_last_layer_input)]
                    self.module[module_name]["input"] = module_input
                else:  # Input innerhalb eines Moduls, dann erstelle Strings von Neuronen im vorherigen Layer
                    input_tmp = [f"{module_name}_{layer_pos + 1}_{n_list_comp}" for n_list_comp in range(self.module[module_name]["hidden"][layer_pos + 1])]

                # Nach der Erstellung des Inputs, werden jetzt die konkreten Neuronen erstellt, initiiert und in das self.model - dict und das self.module[key] - dict eingefügt
                # TODO: Wahrscheinlich besser als eigene Schleife.
                for n_tmp in range(layer_size):
                    neuron_name_tmp = f"{module_name}_{layer_pos}_{n_tmp}"
                    neuron_tmp = Neuron(name=neuron_name_tmp)  # ! TODO: in Konstruktor packen?
                    if layer_pos == 0:  # TODO: temporär, um letzten Layer mit id als act auszustatten!
                        args_tmp = copy.copy(self.args_model)
                        args_tmp["act"] = "id"
                        neuron_tmp.init_act(args_model=args_tmp)
                    else:
                        neuron_tmp.init_act(args_model=self.args_model)
                    neuron_tmp.input_keys = copy.copy(input_tmp)
                    # neuron_tmp.init_weights(args=args_weight_init)

                    self.model[neuron_name_tmp] = neuron_tmp
                    self.module[module_name]["neurons"].append(neuron_tmp)
                    self.module[module_name]["hidden_layer"][layer_pos].append(neuron_name_tmp)

            # Erstellung des Outputs für jedes Modul, welche ebenfalls in self.model und self.module eingefügt werden
            input_tmp = [f"{module_name}_0_{n_list_comp}" for n_list_comp in range(self.module[module_name]["hidden"][0])]
            for l in labels:
                neuron_name_tmp = f"{module_name}_{l}"
                neuron_tmp = Neuron(name=neuron_name_tmp)
                args_tmp = copy.copy(self.args_model)
                # args_tmp["act"] = "sigmoid"
                args_tmp["act"] = "id"  # TODO: Trainingsmodus implementieren!
                # args_tmp["act"] = "relu"
                neuron_tmp.init_act(args_model=args_tmp)
                # neuron_tmp.init_act(args_model=self.args_model)
                neuron_tmp.input_keys = copy.copy(input_tmp)
                neuron_tmp.is_output = True

                self.model[neuron_name_tmp] = neuron_tmp
                self.module[module_name]["output"].append(neuron_tmp)  # ! TODO geht bestimmt kompakter

        # Input des models mit Neuronen (Klasse füllen), also Strings mit den konkreten Neuronen ersetzen
        # output_keys mit Namen fülleb
        for neuron in self.model.values():
            for input_key in neuron.input_keys:
                if input_key in self.features:  # ! TODO: Features haben kein output_keys, könnte hilfreich sein.
                    neuron.input.append(input_key)
                else:
                    neuron.input.append(self.model[input_key])
                    self.model[input_key].output_keys.append(neuron.name)

        # Dokumentation von Fehler der einzelnen Module
        for module_name in self.module.keys():
            self.error_module[module_name] = []

        # Tiefe der Module berechnen
        self.cal_depth_module(key_output=output)

        # self.local_graph()
        self.module_ordering()

    def init_weights(self, args_weight_init):
        # Ausgalagert, wegen normalized Xavier, braucht input_keys
        for neuron in self.model.values():
            neuron.init_weights(args=args_weight_init)

    def module_ordering(self):
        self.module_order = [tuple([key, m["depth"]]) for key, m in self.module.items()]
        self.module_order.sort(key=lambda x: x[1], reverse=True)

    #cal_depth_module bezieht alle inputs ein. Daher kann ein Modul häufiger ausgewertet werden. Das muss sein, da ein Modul von mehrern anderen Modulen erreicht werden kann
    # und ein Weg kürzer als der andere sein kann.
    def cal_depth_module(self, key_output):
        queue = [key_output]
        self.module[key_output]["depth"] = 0
        while len(queue) != 0:
            q_pop = queue.pop(0)

            for module_input in self.module[q_pop]["input"]:
                module_tmp = self.module[q_pop]
                if module_input in self.features or module_input in queue: # module_input continue, richtig? auch bei mehreren pathways, die zum gleichen Ziel führen
                    continue

                if self.module[module_input]["depth"] == None:
                    self.module[module_input]["depth"] = self.module[q_pop]["depth"] + 1
                else:
                    if self.module[module_input]["depth"] > self.module[q_pop]["depth"] + 1:  # kürzere depth nehmen, um kürzeste Tiefe bei Schleife zu nehmen, kommt im Moment nicht vor
                        self.module[module_input]["depth"] = self.module[q_pop]["depth"] + 1

                queue.append(module_input)

        for module_key in self.module.keys():
            self.cal_depth_neuron(module=self.module[module_key])

    def cal_depth_neuron(self, module):
        for layer_depth, neurons in module["hidden_layer"].items():
            for neuron in neurons:
                self.model[neuron].depth = [module["depth"], layer_depth]

    def set_optimiser(self, lr):
        # trainierbare Variablen einsammeln
        self.var_trainable = []
        for key, neuron in self.model.items():
            #self.var_trainable.append(neuron.weights)
            #self.var_trainable.append(neuron.bias)
            self.var_trainable.append(neuron.weights_bias)

        #! mehr als 1 Optimizer implementieren
        self.optimizer = torch.optim.Adam(self.var_trainable, lr=lr)

    def set_loss(self, args):
        self.loss_type = args["loss"]
        #! oder mit Dict? kp was besser ist.
        if(self.loss_type == "MSELoss"):
            self.loss_function = torch.nn.MSELoss()
        if(self.loss_type == "CrossEntropyLoss"):
            if args["disable_weights"]:
                self.loss_function = torch.nn.CrossEntropyLoss()
            else:
                self.loss_function = torch.nn.CrossEntropyLoss(torch.Tensor(args["weights"]))
        if(self.loss_type == "BCELoss"):
            self.loss_function = torch.nn.BCELoss()

    def loss_calculate(self, y_pred, y_true):
        if(self.loss_type == "CrossEntropyLoss"): #! Vielleicht gibt es noch mehr ausnahmen, dann macht Struktur mit dict vielleicht mehr Sinn oder mehr Sonderfälle
            y_true_tmp = torch.max(y_true, 1)[1]
            loss = self.loss_function(y_pred, y_true_tmp)
        else:
            loss = self.loss_function(y_pred, y_true)
        return loss

    def set_args(self, args):
        for key, item in args.items():
            setattr(self, key, item)

    # Wird benötigt, um Input und Output-Tensoren der Neuronen zu reseten, damit sie für Autograd nicht 2-mal aufgerufen werden.
    def reset_tensor(self):
        for n in self.model.values():
            n.input_tensor = None
            n.output_tensor = None

    def predict_and_eval(self, dataset, module_name=None, reset_tensor=True):
        if module_name == None:
            module_name = self.output

        # Predict hole Dataset
        #x_train, y_train = dataset.tensor_output()
        x_train, y_train = dataset[0:len(dataset)]
        x_train = {key: item.unsqueeze(1) for key, item in x_train.items()}
        y_train = {key: item.unsqueeze(1) for key, item in y_train.items()}
        self.x_batch = x_train

        y_pred = self.prediction_module(x=x_train, module_name=module_name)
        if reset_tensor:
            self.reset_tensor()

        # Reshape von label. Anpassung an Shape von Vorhersagen
        y_true = self.reshape_y_true(y=y_train)

        # Berechnung von Loss
        self.loss = self.loss_calculate(y_pred, y_true)

        #acc, f1_score, sensitivity, specificity, mcc = g_eval.scores(y_pred=y_pred, y_true=y_true)
        if self.components["metric"] != None:
            metrics = self.components["metric"].component_apply(y_pred=y_pred, y_true=y_true)
        else:
            metrics = None

        return float(self.loss.detach().numpy()), y_pred, y_true, metrics

    # Predict hole Dataset, modulewise
    def predict_and_eval_modulwise(self, dataset, ep):
        for modul_name in self.module.keys():
            eval = self.predict_and_eval(dataset=dataset, module_name=modul_name)
            self.error_module[modul_name].append(eval[0])  # eval[0] ist loss

        self.error_module_ep.append(ep)

    def predict_and_eval_dataset_store_metrics(self, dataset, name): #TODO: besserer Name
        loss, y_pred_softmax, y_true, metrics = self.predict_and_eval(dataset=dataset)
        self.running_loss[name].append(loss)  # TODO: Variable machen, umterschiedliche Metriken erlauben
        self.running_acc[name].append(metrics["acc"]) #acc, f1_score, sensitivity, specificity, mcc,
        self.running_f1[name].append(metrics["f1"])
        self.running_sens[name].append(metrics["sens"])
        self.running_spec[name].append(metrics["spec"])
        self.running_mcc[name].append(metrics["mcc"])
        self.y_pred_last[name] = y_pred_softmax
        self.y_true_last[name] = y_true

    # Vorhersage eines Moduls
    def prediction_module(self, x, module_name):
        pred = []
        for output_tmp in self.module[module_name]["output"]:
            #pred.append(self.model[output_tmp].forward(X=x))
            pred.append(output_tmp.forward(X=x))

        # Zusammenfügen der Vorhersagen, um Loss berechne zu können. Eigentlich nur reshape nutzen.
        """pred_cat = pred[0]
        for item in pred[1:]:  #? Funktioniert das auch bei mehr als 2 Labeln?
            pred_cat = torch.cat((pred_cat, item), 1)"""

        pred = torch.cat(pred, dim=1)

        if not self.training_mode: #https://stackoverflow.com/questions/47120680/why-cant-the-output-of-the-network-go-through-a-softmax-when-using-softmax-cros
            softmax = torch.nn.Softmax(dim=1)
            pred = softmax(pred)

        return pred

    def reshape_y_true(self, y):
        y_cat = y[self.label_list[0]]
        for label_tmp in self.label_list[1:]:
            y_cat = torch.cat((y_cat, y[label_tmp]), 1)
        return y_cat

    def model_to_device(self, device="cpu"):
        for key, neuron in self.model.items():
            neuron.to(device)
            neuron.share_memory()

    def train_multiloss(self):  # https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch
        # exp_class = util_exp.Exp_class()
        args_train_loader = {'batch_size': self.batch_size,
                             'shuffle': True,
                             'num_workers': 0}

        train_loader = data.DataLoader(dataset=self.dataset_train, **args_train_loader)
        self.running_loss = {"train": [], "val": [], "test": []} #TODO: in func auslagern
        self.running_loss_global = []#{"train": [], "val": [], "test": []}
        self.running_acc = {"train": [], "val": [], "test": []}
        self.running_f1 = {"train": [], "val": [], "test": []}
        self.running_mcc = {"train": [], "val": [], "test": []}
        self.running_sens = {"train": [], "val": [], "test": []}
        self.running_spec = {"train": [], "val": [], "test": []}
        self.y_pred_last = {"train": [], "val": [], "test": []}
        self.y_true_last = {"train": [], "val": [], "test": []}

        grad_saved = {}  # Variable, um alle Gradienten zu speichern.
        for i_epoch in range(self.epochs):
            self.training_mode = True
            print("epoch:" + str(i_epoch + 1) + "/" + str(self.epochs))
            time_batch = [time.time(), 0]  # start batch, time last post
            # exp_class.set_class_var(var={key: None for key in self.model.keys()}, name="grad")
            grad_saved_tmp = {} #TODO: noch nicht benutzt, nötig für genaue Analyse von Gradienten. Verhalten während des Trainings.
            running_loss_global_batch = []
            for i_batch, [x_batch, y_batch] in enumerate(train_loader):
                util.print_batch_computation_time(time_batch=time_batch, len_dataset=len(self.dataset_train), i_batch=i_batch, batch_size=self.batch_size)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                # Prediction with multiple labels. Notiz:https://machinelearningmastery.com/multi-label-classification-with-deep-learning/
                grad = {key: {} for key in self.model.keys()}  # In grad werden alle Gradienten auch in Abhängigkeit der Tiefe der Module gespeichert.
                # Iterativ alle Fehler von allen Modulen berechnen, module_output_name steht für das Modul, bei dem in dem Moment der Fehler berechnet wird.
                # Reihendfolge wichtig, damit nicht 2-mal auf Tensoren zugegriffen wird, gibt sonst Probleme bei Autograd
                loss_glob = torch.tensor(0)
                for module_output_name, module_output_depth in self.module_order:  # ! TODO: eigene Parallelisierung schreiben :/
                    y_pred = self.prediction_module(x=x_batch, module_name=module_output_name)

                    # Reshape von label. Anpassung an Shape von Vorhersagen
                    y_true = self.reshape_y_true(y=y_batch)
                    self.reset_tensor()

                    # Berechnung von Loss, extra Funktion, weil nicht alle Fehlerfunktionen von Pytorch one-hot encoding annehmen.
                    self.loss = self.loss_calculate(y_pred=y_pred, y_true=y_true)

                    if self.components["penalty"] != None:
                        self.components["penalty"].component_apply(module_output_name)

                    rel = self.components["multiloss"].component_apply(depth=module_output_depth, ep=i_epoch)
                    self.loss = torch.multiply(self.loss, rel)

                    loss_glob = torch.add(loss_glob, self.loss)  # TODO: Hier abhängig von Tiefe loss gewichten

                    # Optimierungsschritt
                    self.loss.backward(retain_graph=True)

                    #self.local_grad_saved(grad=grad, module_name=module_output_name, module_depth=module_output_depth)

                    #self.optimizer.zero_grad()
                # exp_class.add_grads(copy.copy(grad))
                self.loss = loss_glob
                running_loss_global_batch.append(loss_glob.detach().numpy())
                self.loss.backward(retain_graph=True)
                #self.local_grad_combination(grad=grad, ep=i_epoch)
                self.optimizer.step()
                self.optimizer.zero_grad()

            #running_loss_global_batch = torch.mean(torch.tensor(running_loss_global_batch)).detach().numpy()
            self.running_loss_global.append(float(np.mean(running_loss_global_batch)))
            print(f"\r \t time passed for ep {round((time.time() - time_batch[0]) / 60, 2)} min")
            grad_saved[str(i_epoch)] = copy.copy(grad_saved_tmp)

            # exp_class.set_class_var(var=i_epoch, name="i_epoch")
            # exp_class.boxplot()

            self.training_mode = False
            # Train loss
            #self.predict_and_eval_modulwise(dataset=self.dataset_train, ep=i_epoch)  # Fehler von jedem Modul wird in einer Klassenvariable gespeichert.
            self.predict_and_eval_dataset_store_metrics(dataset=self.dataset_train, name="train")
            print(f"Train-dataset:\tLoss: {round(self.running_loss['train'][-1], 4)}, Acc: {round(self.running_acc['train'][-1], 2)}, F1: {round(self.running_f1['train'][-1], 2)}, Global loss: {round(self.running_loss_global[-1], 4)}")

            # Validation loss
            self.predict_and_eval_dataset_store_metrics(dataset=self.dataset_validation, name="val")
            print(f"Val-dataset:\tLoss: {round(self.running_loss['val'][-1], 4)}, Acc: {round(self.running_acc['val'][-1], 2)}, F1: {round(self.running_f1['val'][-1], 2)}")

            # Test loss
            if (len(self.dataset_test) != 0):
                self.predict_and_eval_dataset_store_metrics(dataset=self.dataset_test, name="test")
                print(f"Test-dataset:\tLoss: {round(self.running_loss['test'][-1], 4)}, Acc: {round(self.running_acc['test'][-1], 2)}, F1: {round(self.running_f1['test'][-1], 2)}, MCC: {round(self.running_mcc['test'][-1], 2)}")

            # Speichern des besten models
            if self.components["best_model"] != None:
                self.components["best_model"].component_apply(ep=i_epoch)

            # Anwendung von early_stopping, vielleicht mehrere Kriterien notwendig
            if self.components["early_stopping"] != None:
                if self.components["early_stopping"].component_apply():
                    break

        if self.components["best_model"] != None:
            if (self.components["best_model"].epoch < self.epochs): #TypeError -> zu kurze Trainingszeit, noch nicht alle Module wurden trainiert
                #self = self.components["best_model"].best_model_load()
                self.model, self.module = self.components["best_model"].best_model_load()

        if self.components["local_grad"] != None:
            self.components["local_grad"].export() #in_modules=False

        #util.export_saved_grad(grad=grad_saved)
        del self.loss
        return {"loss": self.running_loss,
                "acc": self.running_acc,
                "f1": self.running_f1,
                "sens": self.running_sens,
                "spec": self.running_spec,
                "mcc": self.running_spec,
                "y_pred": self.y_pred_last,
                "y_true": self.y_true_last,
                "loss_global": self.running_loss_global}

    # Berechnet den Abstand der Tiefe der Module, also wie weit das Modul, an dem der Fehler berechnet wird und dem Modul, in dem der Gradient berechnet wurde. Also depth[0] für die Neuronen
    def local_grad_saved(self, grad, module_name, module_depth):
        for name_neuron in self.local_graph[module_name]:
            neuron = self.model[name_neuron]
            #grad_tmp = neuron.weights.grad.detach().numpy()
            grad_tmp = neuron.weights_bias.grad.detach().numpy()
            diff_depth = abs(neuron.depth[0] - module_depth)
            if diff_depth not in grad[name_neuron]:
                grad[name_neuron][diff_depth] = []
            grad[name_neuron][diff_depth].append(copy.copy(grad_tmp))

    def local_grad_combination(self, grad, ep):
        for neuron_name in grad.keys():
            list_iter = list(grad[neuron_name].keys())  # Anzahl der Gradienten, der für dieses Neuron existiert, 0 ist der lokale Gradient, 1 usw. aus höherer Tiefe

            grad_iter = grad[neuron_name]
            for d in grad_iter.keys():
                grad_iter[d] = np.mean(grad_iter[d], axis=0)

    def local_grad_combination_tmp(self, grad, ep):
        for neuron_name in grad.keys():
            list_iter = list(grad[neuron_name].keys())
            list_iter.sort()
            grad_new = copy.copy(grad[neuron_name][list_iter[0]][0])  # Gradient der mit dem Modul als output berechnet wurde. Initialer Gradient!
            #grad_init = copy.copy(grad[neuron_name][list_iter[0]][0])  #Debug
            for depth_tmp in list_iter[1:]:
                grad_tmp = np.mean(grad[neuron_name][depth_tmp], axis=0)  # Mittelwert, falls mehrere Gradienten die gleiche Entfernung haben.
                diff_grad = (grad_new - grad_tmp)  # grad_new

                # Anwendung von early_stopping, vielleicht mehrere Kriterien notwendig
                rel = self.components["multiloss"].component_apply(depth=depth_tmp, ep=ep)

                diff_grad = diff_grad * rel
                grad_new = grad_new - diff_grad

            #self.model[neuron_name].weights.grad = torch.tensor(copy.copy(grad_new))
            self.model[neuron_name].weights_bias.grad = torch.tensor(copy.copy(grad_new))

            if self.components["local_grad"] != None:  # Grad vorher nachher? Anweichung (Einfluss) speichern und dann ploten
                self.components["local_grad"].component_apply(neuron_name=neuron_name, grad_multiloss=copy.copy(grad_new), grad_init=grad[neuron_name][list_iter[0]][0], ep=ep)

    def print_grad(self):
        for item in self.var_trainable:
            print(item._grad)

    def disable_bias_func(self):
        if self.args_controler["disable_bias"]:
            for key in self.model.keys():
                self.model[key].disable_bias = True
                self.model[key].bias = torch.tensor([0], requires_grad=False)

        if self.args_controler["disable_first_layer_bias"]:
            for module in self.module.values():
                layer = [n for n in module["hidden_layer"][len(module["hidden"]) - 1]]
                for key_neuron in layer:
                    self.model[key_neuron].disable_bias = True
                    self.model[key_neuron].bias = torch.tensor([0], requires_grad=False)

        if self.args_controler["disable_last_layer_bias"]:
            for module in self.module.values():
                layer = [n for n in module["hidden_layer"][0]]
                for key_neuron in layer:
                    self.model[key_neuron].disable_bias = True
                    self.model[key_neuron].bias = torch.tensor([0], requires_grad=False)

        if self.args_controler["disable_output_weights_bias"]:
            for module in self.module.values():
                for neuron in module["output"]:
                    neuron.disable_bias = True
                    neuron.bias = torch.tensor([0], requires_grad=False)

    def local_graph(self, modules):
        local_graph_tmp = {item[0]: [input_tmp for input_tmp in item[1] if input_tmp not in self.features] for item in modules}
        self.local_graph_modules = {}
        for key in local_graph_tmp.keys():
            self.local_graph_modules[key] = list(self.recursive_local_graph_modules(local_graph_tmp, key))

        #local_graph_tmp = self.local_graph
        self.local_graph = {key: [] for key in self.local_graph_modules.keys()}
        for key, item in self.local_graph_modules.items():
            for i in item:
                neurons = []
                for n in self.module[i]["hidden_layer"].values():
                    neurons = neurons + n
                self.local_graph[key] = self.local_graph[key] + neurons

    def recursive_local_graph_modules(self, local_graph_tmp, module_tmp):
        new_modules = set([module_tmp])
        for key_m in local_graph_tmp[module_tmp]:
            new_modules = new_modules.union(self.recursive_local_graph_modules(local_graph_tmp=local_graph_tmp, module_tmp=key_m))
        return new_modules
