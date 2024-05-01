import os
from src import eval_cv, util_cv
import pandas as pd
pd.options.mode.chained_assignment = None

# Pfad of crossvalidation
args = {"name_model": "logic"}
args["path_model"] = f"output/{args['name_model']}"
args["path_eval"] = f"output/{args['name_model']}/evaluation"
util_cv.create_directory(directory_new=args["path_eval"])

path_hierarchy = f"data/logic/logic_hierarchy.txt"

# Unterordner der Kreuzvalidierung
models = os.listdir(path=f"{args['path_model']}/models")
models.sort()
args["models"] = models

# Model structure
source_structure = f"{args['path_model']}/structure.txt"
features, modules, output = util_cv.import_structure(filename=source_structure)

labels, _ = util_cv.import_data(filename=f"{args['path_model']}/data/label.txt")
labels = labels[list(labels.keys())[0]]
args["label"] = sorted(list(set(labels)))

modules_dict = {}
for module in modules:
    modules_dict[module[0]] = module[1]
args["module_hierarchy"] = modules_dict
args["features"] = features

args["modules_depth"] = util_cv.cal_depth_module(modules=modules_dict, key_output=output, features=features)
args["depth"] = max(list(args["modules_depth"].values())) + 1

# Pfad der Hierarchy
args["rename"] = util_cv.import_pw_names(path=path_hierarchy)
args["ilmn_hsa"] = util_cv.import_ilmn_hsa(path=path_hierarchy)

#lrp_test
args["lrp"] = f"{args['path_eval']}/lrp"
util_cv.create_directory(directory_new=args["lrp"])
# Zusammenfassen und Export der lrp-Bewertungen
lrp_cv = eval_cv.summerize_lrp(args=args, lrp_type_dataset="epsilon_test")

path_boxplot = f"{args['path_eval']}/boxplot"
important_pw = []
for d in lrp_cv:
    important_pw = important_pw + list(lrp_cv[d]["names_neuron"][lrp_cv[d]["over_std_median_ratio"]])
util_cv.create_directory(directory_new=path_boxplot)
eval_cv.summerize_single_predictios(args, lrp_type_dataset="epsilon_test", important_pw=important_pw, path_boxplot=path_boxplot)

path_pw = f"{args['path_eval']}/lrp/pathway"
util_cv.create_directory(directory_new=path_pw)
# Gesonderter Export aller verbundenen Pathways zu wichtigen Pathways
eval_cv.export_per_pw(args=args, importances=lrp_cv, path=path_pw)


# Sammelt und plotet die Werte der besten Modelle
args["best_models"] = f"{args['path_eval']}/best_models"
util_cv.create_directory(directory_new=args["best_models"])
eval_cv.summerize_best_models(args=args)

# Sammelt und plotet den Verlauf der Metriken aller Modelle
args["path_output_metric"] = f"{args['path_eval']}/metric"
util_cv.create_directory(directory_new=args["path_output_metric"])
eval_cv.summerize_metrics(args=args)
eval_cv.summerize_metrics_special(args=args)
eval_cv.summerize_metrics_single(args=args)

# Sammelt und plotet die Roc-Curves aller Modelle
args["path_output_roc"] = f"{args['path_eval']}/roc"
util_cv.create_directory(directory_new=args["path_output_roc"])
eval_cv.ROC_global(args=args)
eval_cv.ROC_single(args=args)

# Sammelt und plotet die confusion_Matrizen aller Modelle
args["path_output_conMat"] = f"{args['path_eval']}/conMat"
util_cv.create_directory(directory_new=args["path_output_conMat"])
eval_cv.summerize_conMat(args=args)

print("done")
