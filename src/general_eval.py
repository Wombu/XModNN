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

    return acc, f1_score, sensitivity, specificity, mcc