import copy

import torch

class Penalty():
    def __init__(self, args, method="L2"):
        self.name_component = "penalty"
        self.c = None

        self.all_weights = args["all_weights"]
        self.multiplikator_all_weights = args["multiplikator_all_weights"]

        self.last_layer_weights = args["last_layer_weights"]
        self.multiplikator_last_layer_weights = args["multiplikator_last_layer_weights"]

        self.first_layer_weights = args["first_layer_weights"]
        self.multiplikator_first_layer_weights = args["multiplikator_first_layer_weights"]


        self.penalty_dict = {"tf": self.tf,
                             "L2": self.L2,
                             "L1": self.L1,
                             "L2_neuron": self.L2_neuron}

        self.method = self.penalty_dict[method]

    def init(self):
        None

    def component_apply(self, module_output_name):
        if self.multiplikator_all_weights:
            weights_selected = self.determine_selected_weights(module_output_name=module_output_name, weight_selection="weights_all")
            self.method(weights_selected=weights_selected, multiplicator=self.multiplikator_all_weights)

        if self.multiplikator_last_layer_weights:
            weights_selected = self.determine_selected_weights(module_output_name=module_output_name, weight_selection="weights_last")
            self.method(weights_selected=weights_selected, multiplicator=self.multiplikator_last_layer_weights)

        if self.multiplikator_first_layer_weights:
            weights_selected = self.determine_selected_weights(module_output_name=module_output_name, weight_selection="weights_last")
            self.method(weights_selected=weights_selected, multiplicator=self.multiplikator_first_layer_weights)

    def determine_selected_weights(self, module_output_name, weight_selection): #TODO: globalen Output anders einschränken? alle outputs einschränken? oder komplett frei, ohne penalty machen?
        if weight_selection == "weights_all":
            weights_selected = copy.copy(self.c.local_graph[module_output_name])
            return weights_selected

        if weight_selection == "weights_last":
            weights_selected = copy.copy(self.c.module[module_output_name]["hidden_layer"][0])
            return weights_selected

        if weight_selection == "weights_fist":
            weights_selected = copy.copy(self.c.module[module_output_name]["hidden_layer"][-1])
            return weights_selected

    def L2(self, weights_selected, multiplicator):
        # collect trainable variables
        weights = []
        penalty_sum = torch.tensor(0)
        #for key_neuron in self.c.model.keys():
        for key_neuron in weights_selected:
            #weights.append(torch.sum(torch.pow(self.c.model[key_neuron].weights_bias[:-1], 2)))  # Fehler pro Neuron
            penalty_sum = torch.add(penalty_sum, torch.sum(torch.pow(self.c.model[key_neuron].weights_bias[:-1], 2)))
            #weights.append(self.c.model[key_neuron].weights_bias[:-1])

        """sum_2 = torch.stack(weights, dim=0)  # https://www.pinecone.io/learn/regularization-in-neural-networks/
        sum_2 = torch.sum(sum_2) #TODO: Fehler beim Penalty?
        sum_2 = torch.sum(sum_2)
        sum_2 = torch.sqrt(sum_2)"""
        self.c.loss = torch.add(self.c.loss, torch.multiply(penalty_sum, multiplicator))

    def L2_neuron(self, weights_selected, multiplicator):
        weights = []
        #penalty_sum = torch.tensor(0)
        #for key_neuron in self.c.model.keys():
        for key_neuron in weights_selected:
            weights.append(torch.sum(torch.pow(self.c.model[key_neuron].weights_bias[:-1], 2)))  # Fehler pro Neuron
            #penalty_sum = torch.add(penalty_sum, torch.sum(torch.pow(self.c.model[key_neuron].weights_bias[:-1], 2)))

            #weights.append(self.c.model[key_neuron].weights_bias[:-1])

        sum_2 = torch.stack(weights, dim=0)  # https://www.pinecone.io/learn/regularization-in-neural-networks/
        sum_2 = torch.sum(sum_2)
        #sum_2 = torch.sqrt(sum_2)
        self.c.loss = torch.add(self.c.loss, torch.multiply(sum_2, multiplicator))

    def tf(self): #TODO: noch tf
        weights = []
        for key_neuron in self.c.model.keys():
            #weights.append(torch.sum(torch.pow(self.c.model[key_neuron].weights, 2)))
            weights.append(torch.sum(torch.pow(self.c.model[key_neuron].weights_bias[:-1], 2)))

        sum_2 = torch.stack(weights, dim=0)
        sum_2 = torch.sum(sum_2)
        sum_2 = torch.divide(sum_2, 2)
        #sum_2 = torch.abs(sum_2)
        self.c.loss = torch.add(self.c.loss, torch.multiply(sum_2, self.multiplicator))

    def L1(self, weights_selected, multiplicator): #TODO: noch L1
        penalty_sum = torch.tensor(0)
        for key_neuron in weights_selected:
            penalty_sum = torch.add(torch.sum(torch.abs(self.c.model[key_neuron].weights)), penalty_sum)

        self.c.loss = torch.add(self.c.loss, torch.multiply(penalty_sum, multiplicator))
