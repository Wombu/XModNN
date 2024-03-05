import torch

# TODO: übergeordnete Componentenklasse?
class Multiloss_tmp():
    def __init__(self, args):
        self.name_component = "multiloss"
        self.c = None
        self.depth_bool = args["depth"]
        self.r = args["r"]
        self.raise_r = args["raise"]
        self.raise_thresholds = args["raise_thresholds"]
        self.raise_per_epoch = args["raise_per_epoch"]
        self.exponent = args["exponent"]
        self.ep = 0  # erhöht sich bei jedem Aufruf

    def init(self):
        None

    def component_apply(self, depth, ep):
        if self.raise_r:
            r = self.raise_r_iterative(ep=ep)
        else:
            r = self.r
        if self.depth_bool:
            r = self.depth_apply(r=r, depth=depth)
        return r

    def depth_apply(self, r, depth):
        return r ** depth

    def raise_r_iterative(self, ep):
        self.r = self.r + (self.raise_per_epoch * ep) ** self.exponent
        if self.r > self.raise_thresholds[1]:
            self.r = self.raise_thresholds[1]
        if self.r < self.raise_thresholds[0]:
            self.r = self.raise_thresholds[0]
        return self.r

import torch

# TODO: übergeordnete Componentenklasse?
class Multiloss():
    def __init__(self, args):
        self.name_component = "multiloss"
        self.c = None

        self.threshold_epoch = args["threshold_epoch"]
        self.multiloss_weights = args["multiloss_weights"]
        self.ep = 0  # erhöht sich bei jedem Aufruf

    def init(self):
        None

    # reverse accumulation
    """The basic idea through reverse accumulation is to begin at the output node of the computational graph,
and then consider each path in the computational graph from the output node to the input nodes.
• We follow two simple rules: (1) add together different path accumulations and (2) multiply local
derivatives along each path. [Portland State University, Skript](https://web.pdx.edu/~arhodes/cv3.pdf)"""
    #https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html
    # https://openreview.net/pdf?id=BJJsrmfCZ [Paszkeat al., Automatic differentiation in PyTorch, NIPS 2017 Workshop Autodiff Decision Program Chairs]
    def component_apply(self, depth, ep):
        #ep = 6
        """depth_rel = {0: 1, 1: 1, 2: 1, 3: 1}
        return depth_rel[depth]"""

        if ep < self.threshold_epoch[0]:
            depth_rel = {0: 0, 1: 0, 2: 0, 3: self.multiloss_weights[3]}
            return depth_rel[depth]

        if ep < self.threshold_epoch[1]:
            depth_rel = {0: 0, 1: 0, 2: self.multiloss_weights[2], 3: self.multiloss_weights[3]}
            return depth_rel[depth]

        if ep < self.threshold_epoch[2]:
            depth_rel = {0: 0, 1: self.multiloss_weights[1], 2: self.multiloss_weights[2], 3: self.multiloss_weights[3]}
            return depth_rel[depth]

        if ep >= self.threshold_epoch[2]:
            depth_rel = {0: self.multiloss_weights[0], 1: self.multiloss_weights[1], 2: self.multiloss_weights[2], 3: self.multiloss_weights[3]}
            return depth_rel[depth]
