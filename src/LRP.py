import copy
import numpy as np
import torch
import time
#import pandas as pd

class NeuronLRP:
    #def __init__(self, name, weights=None, input_keys=[], bias=0,  activation=None, is_feature=False, depth=None):
    def __init__(self, name, neuron, is_feature):
        self.name = name

        self.R_j = None
        self.R_residual = None
        self.z_j = None
        self.inc_importance_Rij = {}

        self.diasable_bias = neuron.disable_bias

        if is_feature:
            self.is_feature = True  # is feature unnötig, einfach länge von input keys überprüfen, sieht aber schöner aus.
            self.input_keys = []
        else:
            self.is_feature = False
            self.input_keys = copy.copy(neuron.input_keys)

            #self.weights = list(neuron.weights.detach().numpy())
            #self.weights = neuron.weights.detach()
            self.weights_bias = neuron.weights_bias.detach()
            #self.importance_per_weight = [None for i in range(len(self.weights))]
            self.activation = None  # from previeous Neurons

            #self.bias = neuron.bias.detach()#.numpy()
            self.depth = neuron.depth

    def transfer_values(self, neuron):
        try:
            self.input_tensor = neuron.input_tensor.detach()
            self.output_tensor = neuron.output_tensor.detach()
        except AttributeError:
            self.input_tensor = None
            self.output_tensor = None

    def import_importance(self, imp):
        if self._importance is None:
            self._importance = imp
        else:
            self._importance = self._importance + imp

class LRP:
    def __init__(self, controler):
        self.controler = controler  # Wahrscheinlich nicht notwendig
        self.datasets = self.import_datsets(controler)
        self.node_lrp = self.convert_neurons_to_neuronLRP(controler)

        self.lrp_type = {"epsilon": self.lrp_epsilon,
                         "gamma": self.lrp_gamma,
                         "ab": self.lrp_ab,
                         "basic": self.lrp_basic}

    def import_datsets(self, controler): #unnötig, ist in Controler
        dataset_dict = {}
        dataset_dict["train"] = controler.dataset_train
        dataset_dict["validation"] = controler.dataset_validation
        dataset_dict["test"] = controler.dataset_test
        return dataset_dict

    def convert_neurons_to_neuronLRP(self, controler):
        feature_set = set(controler.features)
        neuron_set = set(list(controler.model.keys()))

        # Dict mit allen Neuronen und Features erstellen für die eine Importance berechnet wird
        node_LRP = {}

        for key, neuron in self.controler.model.items():
            node_LRP[key] = NeuronLRP(name=key, neuron=neuron, is_feature=False)

        for key in feature_set:
            node_LRP[key] = NeuronLRP(name=key, neuron=neuron, is_feature=True)

        #  inc_importance für lrp Berechnung erstellen, zuordnung wird so später einfacher
        for key, neuron in node_LRP.items():
            for input_key in neuron.input_keys:
                node_LRP[input_key].inc_importance_Rij[neuron.name] = None

        return node_LRP

    def eval_dataset(self, dataset=None, type="lrp_epsilon", lrp_type="lrp"):
        _, pred, true, _ = self.controler.predict_and_eval(dataset=self.datasets[dataset], reset_tensor=False)
        self.pred_transfer_to_node_lrp()
        self.node_lrp_blanc = copy.deepcopy(self.node_lrp)
        if lrp_type == "lrp":
            initial_R = self.controler_lrp(type=type, pred=pred)
        if lrp_type == "clrp":
            initial_R = self.controler_clrp(type=type, pred=pred)
        if lrp_type == "sglrp":
            initial_R = self.controler_sglrp(type=type, pred=pred)

        self.normalise_importances(initial_R=initial_R, lrp_type=lrp_type)

        importances = self.collect_importances(pred=pred)
        self.controler.reset_tensor()

        #self.controler.reset_tensor()
        return importances

    def tensor_indize_max_pred(self, pred):
        v_max, v_indize = torch.max(pred, dim=1)
        tensor_indize = torch.zeros(size=pred.size())
        for l, label in enumerate(self.controler.label_list):
            tensor_indize[:, l][v_indize == l] = 1
        return tensor_indize

    def normalise_importances(self, initial_R, lrp_type):
        ini_R = torch.sum(initial_R, dim=1).unsqueeze(1)

        # Debug: Kontrolle, ob Summen in den layern gleich sind
        """sum_R_key_depth = {}
        for key_depth, key_modules in module_depth.items():
            if key_depth not in sum_R_key_depth:
                sum_R_key_depth[key_depth] = torch.zeros(size=ini_R.size())
            for key_module in key_modules:
                for n_key in self.controler.module[key_module]["hidden_layer"][0]:
                    sum_R_key_depth[key_depth] += self.node_lrp[n_key].R_j.squeeze(1)"""

        for neuron in self.node_lrp.values():
            if neuron.R_j is not None:
                t_tmp = neuron.R_j / ini_R
                neuron.R_j_norm = copy.copy(t_tmp)
                neuron.R_residual = neuron.R_residual / ini_R

    def collect_importances(self, pred):
        tensor_pred = self.tensor_indize_max_pred(pred=pred)
        # certainty thresholds berechnen und mit tensor_pred verbinden
        certainty_level = {}
        certainty_pred_level = {}
        for c in self.certainty_level():
            tensor_certainty = torch.zeros(size=pred.size())
            tensor_certainty[pred > c] = 1
            tensor_pred_certainty = tensor_certainty * tensor_pred

            tensor_certainty = torch.sum(tensor_certainty, dim=1).type(torch.bool)
            tensor_pred_certainty = tensor_pred_certainty.type(torch.bool)
            certainty_level[c] = tensor_certainty
            certainty_pred_level[c] = tensor_pred_certainty

        # Sammlung von allen betrachteten Nodes
        key_imp = []
        for module in self.controler.module.values():
            key_imp += module["hidden_layer"][0]
        key_imp += self.controler.features

        # Berechnen der Werte
        importances = {}
        for key_nodes in key_imp:
            node_tmp = self.node_lrp[key_nodes]
            imp_tmp = self.node_lrp[key_nodes].R_j_norm  # R_j
            imp_res_tmp = self.node_lrp[key_nodes].R_residual

            importances[key_nodes] = []
            importances[f"{key_nodes}_residual"] = []

            for c, tensor_certainty in certainty_level.items():
                importances[key_nodes].append(torch.sum(imp_tmp[tensor_certainty]))
                importances[f"{key_nodes}_residual"].append(torch.sum(imp_res_tmp[tensor_certainty]))

            for c, tensor_pred_certainty in certainty_pred_level.items():
                for l, label in enumerate(self.controler.label_list):
                    importances[key_nodes].append(torch.sum(imp_tmp[tensor_pred_certainty[:, l]]))
                    importances[f"{key_nodes}_residual"].append(torch.sum(imp_res_tmp[tensor_pred_certainty[:, l]]))

        columnames = [f"{c}_sum" for c in certainty_level.keys()]

        return importances, columnames

    def certainty_level(self): #TODO: Geht bestimmt eleganter :D
        certainty_min = round(float(1/len(self.controler.label_list)), 1)
        certainty_level = []
        while certainty_min < 1:
            certainty_level.append(copy.copy(certainty_min))
            certainty_min = round(certainty_min + 0.1, 1)
        certainty_level.append(1)
        return certainty_level

    def pred_transfer_to_node_lrp(self):
        #  Vorhersage eines samples und übertragung von Werte in self.node_lrp
        for key, neuron in self.controler.model.items():
            self.node_lrp[key].transfer_values(neuron)

    def order_neurons(self):
        # Module nach Tiefe ordnen, damit LRP, richtig von hinten nach vorn iterrieren kann
        module_order = [[key, m["depth"]] for key, m in self.controler.module.items()]
        module_order.sort(key=lambda x: x[1], reverse=False)
        # output des netzes anfürgen, da es nicht in module["neurons"] gespeichert ist
        neuron_order = [n.name for n in self.controler.module[self.controler.output]["output"]]  # Könnte in vorherige Schleife ausgelagert werden
        # Neuronen der Module ebenfalls ordnen
        for module_key, depth in module_order:
            neuron_module_keys = [n.name for n in self.controler.module[module_key]["neurons"]]
            neuron_module_keys.sort(key=lambda x: self.controler.model[x].depth[1], reverse=False)
            neuron_order += neuron_module_keys

        return neuron_order

    # Explaining Convolutional Neural Networks using Softmax Gradient Layer-wise Relevance Propagation, https://arxiv.org/pdf/1908.04351.pdf
    # Gradient für jede nach Softmax-Vorhersage
    # Backprob
    # Max (inkl. 0) nehmen
    def controler_sglrp(self, type, pred):
        neuron_order = self.order_neurons()

        #tensor_pred = self.tensor_indize_max_pred(pred)  # Vorhersage
        #tensor_bool_pred = tensor_pred.type(torch.bool)

        # Berechnung aller Relevanzen, iterativ, alsob alle Label nacheinander das vorhergesagte Target wären.
        sglrp_initial_relevances_pertarget = {}
        for target_enum, target_label in enumerate(self.controler.label_list):
            sglrp_initial_relevances_pertarget[target_enum] = torch.zeros(size=pred.size())
            for label_enum, label in enumerate(self.controler.label_list):
                if target_enum == label_enum:
                    R1 = (pred[:, target_enum] * (1 - pred[:, target_enum]))
                    sglrp_initial_relevances_pertarget[target_enum][:, label_enum] = R1
                    # R = y_target * (1 - y_target)
                else:
                    R2 = (-1*pred[:, target_enum] * pred[:, label_enum])
                    sglrp_initial_relevances_pertarget[target_enum][:, label_enum] = R2
                    # R = -y_target * y_label

        # Auswahl der Relevancen der tatsächlich vorhergesagten Targets
        sglrp_initial_relevances = torch.zeros(size=pred.size())
        for pre_enum, pred_tmp in enumerate(pred):
            i_tmp = torch.argmax(pred_tmp)
            sglrp_initial_relevances[pre_enum] = sglrp_initial_relevances_pertarget[int(i_tmp)][pre_enum]

        # erst Relevance für Vorhersage "tensor_bool_pred"
        # Dann Relevance für andere Klassen "tensor_bool_pred_not" # TODO: im Moment nur für 2 Klassen möglich!
        # nicht Vorhergsage Klasse
        #self.node_lrp = copy.deepcopy(self.node_lrp_blanc)
        for l_pos, neuron in enumerate(self.controler.module[self.controler.output]["output"]):
            key_neuron = neuron.name
            test = sglrp_initial_relevances[:, l_pos].unsqueeze(-1)
            self.node_lrp[key_neuron].R_j = test
            self.node_lrp[key_neuron].R_residual = torch.zeros(size=test.size())
            #output_tensor = self.node_lrp[key_neuron].output_tensor
            #self.node_lrp[key_neuron].R_j = torch.multiply(output_tensor, tensor_bool_pred_not[:, l_pos].unsqueeze(-1))


        # LRP anwenden
        time1 = time.time()
        for neuron_key in neuron_order:
            self.lrp(node_key=neuron_key, type=type)

        return sglrp_initial_relevances

    # Understanding Individual Decisions of CNNs via Contrastive Backpropagation, https://arxiv.org/pdf/1812.02100.pdf
    def controler_clrp(self, type, pred):  # TODO: überarbeiten, schöner programmieren + init R_j Bedingung hinzufügen (Steht im sgpaper besser beschrieben)
        neuron_order = self.order_neurons()

        tensor_bool_pred = self.tensor_indize_max_pred(pred)  # Vorhersage
        tensor_bool_pred = tensor_bool_pred.type(torch.bool)
        tensor_bool_pred_not = ~tensor_bool_pred

        # erst Relevance für Vorhersage "tensor_bool_pred"
        # Dann Relevance für andere Klassen "tensor_bool_pred_not" # TODO: im Moment nur für 2 Klassen möglich!
        # nicht Vorhergsage Klasse
        self.node_lrp = copy.deepcopy(self.node_lrp_blanc)
        for l_pos, neuron in enumerate(self.controler.module[self.controler.output]["output"]):
            key_neuron = neuron.name

            output_tensor = self.node_lrp[key_neuron].output_tensor
            self.node_lrp[key_neuron].R_j = torch.multiply(output_tensor, tensor_bool_pred_not[:, l_pos].unsqueeze(-1))
            self.node_lrp[key_neuron].R_residual = torch.zeros(size=output_tensor.size())  # TODO: irgendwie so, dass der residual Wert beim Output 0 ist

        # LRP anwenden
        time1 = time.time()
        for neuron_key in neuron_order:
            self.lrp(node_key=neuron_key, type=type)

        node_lrp_dual = self.node_lrp


        # Vorhergsage Klasse
        self.node_lrp = copy.deepcopy(self.node_lrp_blanc)
        for l_pos, neuron in enumerate(self.controler.module[self.controler.output]["output"]):
            key_neuron = neuron.name

            output_tensor = self.node_lrp[key_neuron].output_tensor
            self.node_lrp[key_neuron].R_j = torch.multiply(output_tensor, tensor_bool_pred[:, l_pos].unsqueeze(-1))

        # LRP anwenden
        for neuron_key in neuron_order:
            self.lrp(node_key=neuron_key, type=type)


        for key in neuron_order + self.controler.features:
            R_j1 = self.node_lrp[key].R_j
            R_j2 = node_lrp_dual[key].R_j
            test = torch.subtract(R_j1, R_j2)
            z = torch.zeros(size=test.size())
            test1 = torch.cat([test, z], dim=1)
            test2, indize = torch.max(test1, dim=1)
            test2 = test2.unsqueeze(dim=1)
            self.node_lrp[key].R_j = test2

        print(f"Zeit für Berechnung von {type} CLRP {time.time() - time1}")

        initial_R = torch.zeros(size=pred.size())
        for l_pos, neuron in enumerate(self.controler.module[self.controler.output]["output"]):
            initial_R[:, l_pos] = neuron.output_tensor
        return initial_R

    def controler_lrp(self, type, pred):
        tensor_bool_pred = self.tensor_indize_max_pred(pred)

        # hier wird die prediction als output definiert, da die softmax-funktion nicht beim output_tensor angewendet wurde
        # Nur Target Class analysieren: Understanding Individual Decisions of CNNs via Contrastive Backpropagation Kap. 3
        # Original ohne Softmax: On Pixel-Wise-Explan... Bach et al. S.28, über (65)
        initial_R = torch.zeros(size=pred.size())
        for l_pos, neuron in enumerate(self.controler.module[self.controler.output]["output"]):
            key_neuron = neuron.name
            # Ein Output, ohne Softmax
            output_tensor = self.node_lrp[key_neuron].output_tensor
            self.node_lrp[key_neuron].R_j = torch.multiply(output_tensor, tensor_bool_pred[:, l_pos].unsqueeze(-1))
            self.node_lrp[key_neuron].R_residual = torch.zeros(size=output_tensor.size())

            initial_R[:, l_pos] = torch.multiply(output_tensor, tensor_bool_pred[:, l_pos].unsqueeze(-1)).squeeze(1) #TODO: Aufräumen

        neuron_order = self.order_neurons()

        # LRP anwenden
        time1 = time.time()
        for neuron_key in neuron_order:
            self.lrp(node_key=neuron_key, type=type)
        print(f"Zeit für Berechnung von {type} LRP {time.time() - time1}")

        return initial_R

    def lrp(self, node_key, type):
        node = self.node_lrp[node_key]
        # R_j muss bereits fertig aufsummiert worden sein, da ich das Netz vom Output systematisch rückwärts laufen.
        # In diese Methode muss R_j auf die Gewichte und damit auf die Herkunftsneuronen aufgeteilt werden. -> Importance per weight/per node
        # Bei der Aufteilung wird der Input-Wert, sowie das Gewicht selber nach der lrp-Methode (epsilon etz) berücksichtigt.
        # Danach werden die berechneten Relevance auf die entsprechenden Neuronen zurückgerechnet.
        # R_j initial auf 0 setzen.
        # bei node.is_feature kann abgebrochen werden.

        # If node is Feature, Ende der Berechnung an dieser Stelle
        if node.is_feature:
            return []  # TODO: Output überprüfen, lieber None?

        # Importance für einkommende Neuronen berechnen
        R_ij, R_residual = self.lrp_type[type](node=node)

        # Aufsummieren der Bewertungen die von den output Neuronen weitergegebenen Bewertungen für ein spezielles Neuron
        R_residual = copy.copy(R_residual/len(node.input_keys)).unsqueeze(-1)
        for i, key in enumerate(node.input_keys):
            if self.node_lrp[key].R_j == None:
                self.node_lrp[key].R_j = R_ij[:, i].unsqueeze(-1) #+ R_residual  # TODO: dis/enable residual transfer
                self.node_lrp[key].R_residual = R_residual
            else:
                self.node_lrp[key].R_j = self.node_lrp[key].R_j + R_ij[:, i].unsqueeze(-1) #+ R_residual  # TODO: dis/enable residual transfer
                self.node_lrp[key].R_residual = self.node_lrp[key].R_residual + R_residual

            self.node_lrp[key].inc_importance_Rij[node.name] = R_ij[:, i].unsqueeze(-1)

    # On Pixel-wise Explanations for Non-Linear Classifier Decisions by Layer-wise Relevance Propagation, Bach et al.
    def lrp_basic(self, node):
        z_ij = torch.multiply(node.input_tensor, node.weights_bias)
        z_j = torch.sum(z_ij, dim=1)
        z_j = z_j.unsqueeze(-1) #TODO: squeeze nötig?

        R_ij = torch.divide(z_ij, z_j)
        R_ij = torch.multiply(R_ij, node.R_j)

        return R_ij[:, 0:-1], R_ij[:, -1]

    # On Pixel-wise Explanations for Non-Linear Classifier Decisions by Layer-wise Relevance Propagation, Bach et al.
    def lrp_epsilon(self, node):
        z_ij = torch.multiply(node.input_tensor, node.weights_bias)
        z_j = torch.sum(z_ij, dim=1)
        z_j = z_j.unsqueeze(-1) #TODO: squeeze nötig?

        z_j_sign = torch.sign(z_j)
        epsilon = torch.multiply(z_j_sign, 0.1)
        z_j = torch.add(z_j, epsilon)

        R_ij = torch.divide(z_ij, z_j)
        R_ij = torch.multiply(R_ij, node.R_j)

        R_residual = node.R_j * ((node.weights_bias[-1] + epsilon) / (z_j))

        return R_ij[:, 0:-1], R_residual.squeeze(1)

    # TODO: zu tun ;D
    def lrp_ab(self, z_ij, R_j): #! TODO: wie die anderen definieren, dann noch gamma implementieren
        # ab
        """z_ij_pos = torch.sign(z_ij)
        z_ij_pos[z_ij_pos == -1] = 0
        z_j_pos = torch.sum(torch.multiply(z_ij, z_ij_pos), dim=1)

        z_ij_neg = torch.sign(z_ij)
        z_ij_neg[z_ij_neg == 1] = 0
        z_j_neg = torch.sum(torch.multiply(z_ij, z_ij_neg), dim=1)"""

        z_ij_pos = torch.sign(z_ij)
        z_ij_pos[z_ij_pos == -1] = 0
        z_j_pos = torch.sum(torch.multiply(z_ij, z_ij_pos), dim=1)
        z_j_pos = z_j_pos.unsqueeze(-1)

        z_ij_neg = torch.sign(z_ij)
        z_ij_neg[z_ij_neg == 1] = 0
        z_j_neg = torch.sum(torch.multiply(z_ij, z_ij_neg), dim=1)
        z_j_neg = z_j_neg.unsqueeze(-1)

    #TODO: zu tun ;D
    def lrp_gamma(self, node):
        None


    def set_args(self, args):
        # save args
        for key, item in args.items():
            setattr(self, key, item)
