import torch
import numpy as np
from sklearn import metrics as sk_metrics

class Metrics():
    def __init__(self, args=None):
        self.name_component = "metric"
        self.c = None
        self.args = args
        self.methods = {"acc": self.calculate_acc,
                        #"balenced_acc": self.calculate_balenced_acc(),
                        "f1": self.calculate_f1,
                        "sens": self.calculate_sens,
                        "spec": self.calculate_spec,
                        "mcc": self.calculate_mcc}

    def init(self):
        None

    def component_apply(self, y_pred, y_true): #mit args arbeiten???
        y_pred = np.argmax(y_pred.detach().numpy(), axis=1)
        y_true = np.argmax(y_true.detach().numpy(), axis=1)

        metrics_tmp = {}
        for m in self.args["metrics"]:
            metrics_tmp[m] = self.methods[m](y_true, y_pred)  # wahrscheinlich sogar hier mit args am besten

        return metrics_tmp

    def calculate_acc(self, y_true, y_pred):
        return sk_metrics.balanced_accuracy_score(y_true, y_pred)

    def calculate_f1(self, y_true, y_pred):
        return sk_metrics.f1_score(y_true, y_pred, average="macro")

    def calculate_sens(self, y_true, y_pred):
        return sk_metrics.precision_score(y_true, y_pred, average="macro")

    def calculate_spec(self, y_true, y_pred):
        return sk_metrics.balanced_accuracy_score(y_true, y_pred)

    def calculate_mcc(self, y_true, y_pred):
        return sk_metrics.matthews_corrcoef(y_true, y_pred)


"""    scores["training"]["acc"].append(metrics.balanced_accuracy_score(y_train, y_train_pred))
    scores["training"]["f1"].append(metrics.f1_score(y_train, y_train_pred, average="macro"))
    scores["training"]["spez"].append(metrics.recall_score(y_train, y_train_pred, average='macro'))
    scores["training"]["sens"].append(metrics.precision_score(y_train, y_train_pred, average='macro'))
    scores["training"]["mcc"].append(metrics.matthews_corrcoef(y_train, y_train_pred))



import torch
import numpy as np

# TODO: als Component programmieren!
def scores(y_pred, y_true):
    n = y_pred.size()[0]
    scores = {"TP": 0,
              "FP": 0,
              "TN": 0,
              "FN": 0}

    for i in range(n):
        softmax = torch.nn.Softmax(dim=0)
        y_pred_softmax = list(softmax(y_pred[i]))
        y_pred_index = y_pred_softmax.index(max(y_pred_softmax))

        y_true_list = list(y_true[i].detach().numpy())
        y_true_index = y_true_list.index(max(y_true_list))

        #! Shortcut für Binäres problem mit 2 Labeln (1,0), (0,1) Für mehr Klassen, muss hier multilable F1-Score hin
        if(y_pred_index == 0):
            pred_tmp = 1
        else:
            pred_tmp = 0

        if (y_true_index == 0):
            true_tmp = 1
        else:
            true_tmp = 0

        if pred_tmp == true_tmp:
            if(pred_tmp == true_tmp == 1):
                scores["TP"] += 1
            else:
                scores["TN"] += 1
            continue

        if pred_tmp != true_tmp:
            if(pred_tmp > true_tmp):
                scores["FP"] += 1
            else:
                scores["FN"] += 1
            continue

    acc = (scores["TP"] + scores["TN"])/n
    sensitivity = scores["TP"] / (scores["TP"] + scores["FN"])
    specificity = scores["TN"] / (scores["TN"] + scores["FP"])
    f1_score = scores["TP"] / (scores["TP"] + 0.5*(scores["FP"] + scores["FN"]))
    mcc = ((scores["TP"] * scores["TN"] - scores["FN"] * scores["FN"]) /
           np.sqrt((scores["TP"] + scores["FP"]) * (scores["TP"] + scores["FN"]) * (scores["TN"] + scores["FP"]) * (scores["TN"] + scores["FN"])))
    #https://www.statology.org/matthews-correlation-coefficient-python/

    return acc, f1_score, sensitivity, specificity, mcc"""
