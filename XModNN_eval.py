import copy
import os
import pandas as pd
from src import eval_cv, util_cv

name_model = "logic" # neuroblastoma or logic

# Pfad of crossvalidation
args = {"name_model": name_model,
        "path_model": f"output/{name_model}",
        "path_output": f"output/{name_model}/evaluation"}
util_cv.create_directory(directory_new=args["path_output"])

# Unterordner der Kreuzvalidierung
models = os.listdir(path=f"{args['path_model']}/models")
models.sort()
args["models"] = models

util_cv.create_directory(directory_new=f"output/{args['name_model']}")

# Hierarchy
path_brite = f"data/logic/hierarchy_logic.txt"
#path_brite = f"data/neuroblastoma/hierarchy_neuroblastoma.txt"
annot_brite = util_cv.import_pw_names(path=path_brite)

importance_datasets = []
predictions_datasets = []
for m in args["models"]:
        importance_datasets.append(pd.read_csv(f"{args['path_model']}/models/{m}/LRP/epsilon_test_values/importance_dataset_test.csv", index_col=0))
        predictions_datasets.append(pd.read_csv(f"{args['path_model']}/models/{m}/LRP/epsilon_test_values/predictions_dataset_test.csv", index_col=0))
df_importance = pd.concat(importance_datasets, axis=1)
df_predictions = pd.concat(predictions_datasets, axis=1)

predictions = list(df_predictions.idxmax())
pred_index_dict = {"all": list(range(len(predictions)))}
for i, pred in enumerate(predictions):
    if pred not in pred_index_dict:
        pred_index_dict[pred] = []
    pred_index_dict[pred].append(i)

# Model structure
source_structure = f"{args['path_model']}/structure.csv"
features, neurons, output = util_cv.import_structure(filename=source_structure)

modules_dict = {}
for module in neurons:
    modules_dict[module[0]] = module[1]
module_depth = util_cv.module_depth(modules=modules_dict, key_output=output)

for depth, item in module_depth.items():
    std_threshold = None
    for label, index in pred_index_dict.items():
        df = df_importance.loc[item]
        df_all = copy.copy(df)
        df = df.iloc[:, index]
        df.to_csv(f"{args['path_output']}/local_importance_{label}_{depth}.csv")
        df[df < 0] = 0

        df_dict = {}
        df_dict["median"] = df.median(axis=1)
        df_dict["median_perc"] = (df_dict["median"].abs() / df_dict["median"].abs().sum()) * 100
        df_dict["mean"] = df.mean(axis=1)
        df_dict["std"] = df.std(axis=1)
        df_dict["min"] = df.min(axis=1)
        df_dict["max"] = df.max(axis=1)

        df_new = pd.DataFrame.from_dict(df_dict)

        threshold = df_all.stack().std() + df_all.stack().mean()
        df_new.loc[:, f"over_threshold"] = False
        df_new.loc[df_new["median"] > threshold, f"over_threshold"] = True

        names = [annot_brite[f] if (f in annot_brite) else f for f in df_new.index]
        df_new["names"] = names

        df_new.to_csv(f"{args['path_output']}/importance_{label}_{depth}.csv")

        print(f"depth {depth}, label {label}, done")

print("done")
