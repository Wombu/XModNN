import torch

class Neuron(torch.nn.Module):
    def __init__(self, name=None):
        super().__init__()

        self.name = name

        self.weights = None
        self.bias = None

        #self.input = None  # list (da neuronen als obj gespeichert werden müssen) mit neuronen(obj!) oder features und position
        self.input = []  # list (da neuronen als obj gespeichert werden müssen) mit neuronen(obj!) oder features und position
        self.input_keys = []
        self.output_keys = []
        self.input_tensor = None
        self.output_tensor = None

        self.depth = [float("inf"), float("inf")]  # hilfreich, um output mit 0 zu markieren?
        #self.depth = None  # hilfreich, um output mit 0 zu markieren?
        self.output_pos = None

        self.disable_bias = False

        self.weight_initialiser = {"normal_dist": self.weights_normal_distributed,
                                   "Xavier": self.weights_Xavier,
                                   "normal_Xavier": self.weights_normal_Xavier,
                                   "He": self.weights_He,
                                   "existing": self.weights_existing_initialization}

        self.act_dict = {"tanh": self.tanh,
                         "sigmoid": self. sigmoid,
                         "relu": self.relu}

        self.c = None

    """def prep_input_memory(self, X):  # vllt optimieren, ohne python Liste
        input_tensor = []
        for item in self.input:
            if(item in self.c.x_batch):
                input_tensor.append(self.c.x_batch[item])
            else:
                input_tensor.append(item.forward(X=X))
        #self.input_tensor = torch.stack(tensor)
        #tensor = tensor[0]
        #torch.cat(tensor, dim=1)
        self.input_tensor = torch.cat(input_tensor, dim=1)

        # if key from self.input in X -> Feature, sonst Neuron und aufrufen.
        # output: input tensor"""

    def prep_input(self, X):  # vllt optimieren, ohne python Liste
        input_tensor = []
        for item in self.input:
            if(item in X):
                #test = X[item].squeeze(-1)
                #input_tensor.append(X[item].squeeze(-1))
                input_tensor.append(X[item])
            else:
                if item.output_tensor != None:
                    input_tensor.append(item.output_tensor)
                else:
                    input_tensor.append(item.forward(X=X))

        if self.disable_bias:
            input_tensor.append(torch.zeros(size=input_tensor[0].size()))
        else:
            input_tensor.append(torch.ones(size=input_tensor[0].size()))
        """input_tensor = torch.stack(input_tensor)
        input_tensor = torch.transpose(input_tensor, dim0=0, dim1=1)"""
        input_tensor = torch.cat(input_tensor, dim=1)
        return input_tensor
        #return input_tensor


        # if key from self.input in X -> Feature, sonst Neuron und aufrufen.
        # output: input tensor

    def forward_old(self, X):  # x muss dict/Werte enthalten, die nach hinten gegeben werden können
        self.input_tensor = self.prep_input(X=X)

        sum = torch.multiply(self.weights, self.input_tensor)
        sum = torch.sum(sum, dim=1)
        if not self.disable_bias:
            sum = torch.add(sum, self.bias)
        self.output_tensor = self.act(sum).unsqueeze(1)

        return self.output_tensor

    def forward(self, X):  # x muss dict/Werte enthalten, die nach hinten gegeben werden können
        self.input_tensor = self.prep_input(X=X)

        #sum = torch.multiply(self.weights, self.input_tensor)
        sum = torch.multiply(self.weights_bias, self.input_tensor)
        sum = torch.sum(sum, dim=1)
        """if not self.disable_bias:
            sum = torch.add(sum, self.bias)"""
        self.output_tensor = self.act(sum).unsqueeze(1)

        return self.output_tensor

    def init_weights(self, args={"method": "normal_dist"}):
        self.weights, self.bias = self.weight_initialiser[args["method"]](args)
        #self.weights.requires_grad = True
        #self.bias.requires_grad = True

        self.weights_bias = torch.cat((self.weights, self.bias))
        self.weights_bias.requires_grad = True

    def weights_normal_distributed(self, args):
        weights = torch.distributions.normal.Normal(loc=args["mean"], scale=args["std"]).sample((len(self.input_keys),))
        #bias = torch.distributions.normal.Normal(loc=args["mean"], scale=args["std"]).sample((1,))
        bias = torch.tensor([0.0])
        return weights, bias

    # input in input_keys geändert! Bei umprogrammieren zu modulen mit mehrerern Neuronen
    def weights_Xavier(self, args=None): #Xavier
        # Glorot and Bengio, 2010, Understanding the difficulty of training deep feedforward neural networks
        len_input = torch.tensor(len(self.input_keys))
        threshold = torch.divide(torch.tensor(1), torch.sqrt(len_input))
        weights = torch.distributions.uniform.Uniform(low=-threshold, high=threshold).sample((len(self.input_keys),))
        #bias = torch.distributions.uniform.Uniform(low=-torch.sqrt(torch.tensor(len(self.input_keys))), high=torch.sqrt(torch.tensor(len(self.input_keys)))).sample((1,))
        bias = torch.tensor([0.0])
        return weights, bias

    # Nicht so einfach möglich, da output-neuronen nicht gespeichert sind.
    def weights_normal_Xavier(self, args=None):
        # Glorot and Bengio, 2010
        # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
        len_input = torch.tensor(len(self.input_keys))
        len_output = torch.tensor(len(self.output_keys))
        threshold = torch.divide(torch.sqrt(torch.tensor(6)), torch.sqrt(torch.add(len_input, len_output)))
        #weights = torch.distributions.uniform.Uniform(low=-torch.sqrt(torch.tensor(len(self.input_keys))), high=torch.sqrt(torch.tensor(len(self.input_keys)))).sample((len(self.input_keys),))
        weights = torch.distributions.uniform.Uniform(low=-threshold, high=threshold).sample((len(self.input_keys),))
        #bias = torch.distributions.uniform.Uniform(low=-torch.sqrt(torch.tensor(len(self.input_keys))), high=torch.sqrt(torch.tensor(len(self.input_keys)))).sample((1,))
        bias = torch.tensor([0.0])
        return weights, bias

    def weights_He(self, args=None):
        #He et al.(2015), Delving Deep into Rectifiers:Surpassing Human-Level Performance on ImageNet Classification
        # https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/
        weights = torch.distributions.normal.Normal(torch.tensor(0.0), torch.sqrt(torch.tensor((2 / len(self.input_keys))))).sample((len(self.input_keys),))
        #bias = torch.distributions.normal.Normal(torch.tensor(0.0), torch.sqrt(torch.tensor((2 / len(self.input_keys))))).sample((1,))
        bias = torch.tensor([0.0])
        return weights, bias

    def weights_existing_initialization(self, args):
        weights = torch.Tensor(args["weights"]["weight"])
        bias = torch.Tensor([args["weights"]["bias"]])
        return weights, bias

    def init_act(self, args_model):
        self.act_dict[args_model["act"]]()
        # self.act = torch.nn.LeakyReLU()
        # self.act = torch.nn.ReLU6()
        # self.act = act_c.Sigmoid_positive
        # self.act = act_c.ReLu1

    def tanh(self):
        self.act = torch.nn.Tanh()

    def sigmoid(self):
        self.act = torch.nn.Sigmoid()
    def relu(self):
        self.act = torch.nn.ReLU()
