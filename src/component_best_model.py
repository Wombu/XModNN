import torch
import copy
from src import util

# Deep learning S. 239
class Best_model():
    def __init__(self, args): #! Args input
        self.name_component = "best_model"
        self.c = None
        self.best_model = None
        self.error_val_min = float('inf')
        self.epoch = None
        self.path = args["path"]

        self.acc = {"test": None, "val": None, "train":None}
        self.f1 = {"test": None, "val": None, "train": None}
        self.loss = {"test": None, "val": None, "train": None}
        self.sens = {"test": None, "val": None, "train": None}
        self.spec = {"test": None, "val": None, "train": None}
        self.mcc = {"test": None, "val": None, "train": None}

    def init(self):
        None

    #! Allgemeine Funktion mit args schreiben, damit Klassen später als übergeordnete Klasse component + erben definiert werden können
    def component_apply(self, ep):
        self.best_model_save(ep)

    def best_model_save(self, ep):
        if ep < (self.c.args_multiloss["threshold_epoch"][-1] - 1):  #TODO: try: wenn multiloss nicht da ist? passiert wahrscheinlich nie oder ist 0
            return

        if(self.c.running_loss["val"][-1] < self.error_val_min):
            self.best_model = copy.copy(self.c.model)
            self.best_module = copy.copy(self.c.module)
            self.epoch = ep
            self.error_val_min = copy.copy(self.c.running_loss["val"][-1])

            torch.save(tuple([self.best_model, self.best_module]), f"{self.path}/torch_best_model")

            self.metrics_save(component_dict=self.acc, model_dict=self.c.running_acc)
            self.metrics_save(component_dict=self.f1, model_dict=self.c.running_f1)
            self.metrics_save(component_dict=self.loss, model_dict=self.c.running_loss)
            self.metrics_save(component_dict=self.sens, model_dict=self.c.running_sens)
            self.metrics_save(component_dict=self.spec, model_dict=self.c.running_spec)
            self.metrics_save(component_dict=self.mcc, model_dict=self.c.running_mcc)

            with open(f"{self.path}/metric_values.csv", "w+") as f:
                iter = ["test", "val", "train"]
                f.write(f"metric,test,val,train,\n")
                f.write(f"acc,")
                for a in iter:
                    f.write(f"{self.acc[a]},")
                f.write("\n")
                f.write(f"f1,")
                for a in iter:
                    f.write(f"{self.f1[a]},")
                f.write("\n")
                f.write(f"loss,")
                for a in iter:
                    f.write(f"{self.loss[a]},")
                f.write("\n")
                f.write(f"sens,")
                for a in iter:
                    f.write(f"{self.sens[a]},")
                f.write("\n")
                f.write(f"spec,")
                for a in iter:
                    f.write(f"{self.spec[a]},")
                f.write("\n")
                f.write(f"mcc,")
                for a in iter:
                    f.write(f"{self.mcc[a]},")
                f.write("\n")
                f.write(f"ep,{ep}")

    def metrics_save(self, component_dict, model_dict):
        for key in model_dict.keys():
            try:
                component_dict[key] = copy.copy(model_dict[key][-1])
            except IndexError:  #TODO: Kein Testset, besser lösen
                continue

    def best_model_load(self):
        return torch.load(f"{self.path}/torch_best_model")

    def reset_for_iter(self, path):
        self.best_model = None
        self.error_val_min = float('inf')
        self.epoch = None
        self.path = f"{path}/best_model"
        util.create_directory(self.path)

        self.acc = {"test": None, "val": None, "train":None}
        self.f1 = {"test": None, "val": None, "train": None}
        self.loss = {"test": None, "val": None, "train": None}
        self.sens = {"test": None, "val": None, "train": None}
        self.spec = {"test": None, "val": None, "train": None}
        self.mcc = {"test": None, "val": None, "train": None}
