The eXplainable Modular Neural Network called XModNN allows the identification of important
biomarkers in functional hierarchies to classify diseases and clinical parameters in high
throughput sequencing blood expression datasets. Each module and its connections represent
a pathway or gene within the functional hierarchy. This biologically informed architecture
results in a reduction of parameters and an intrinsic feature selection reinforced by the
weighted multi-loss progressive training enabling the successful classification in less
replicates. This workflow in combination with Layer wise Relevance Propagation ensures a
robust post-hoc explanation of the individual module contribution.

Required packages for python 3.10.9:
- pandas 2.0.3
- torch 1.13.1+cu117
- numpy 1.23.5
- sklearn 1.3.0
- seaborn 0.12.2

Required files in /data:
dataset.csv
  columnames: sample ids
  rownames: featurenames

label.csv
  columnames: sample ids
  rownames: label

structure.csv
Based on this file the internal structure of XModNN is created
F stands for feature, M stands for module, O stands for global network output:
  F,f1,f2,...
  M,modulename1,input1,input2,...
  ...
  M,modulename10,input101,input102,...
  O.output_module,input1001,input1002,...

hierarchy.txt
  This file inherits all individual connections from the functional hierarchy and is used for renaming purposes for the evaluation of XModNN.
  rowwise: hierarchylevel,modulename,pathwayname(\t)hierarchylevel,modulename,pathwayname(\t)...

For all required files we included a test dataset in data/logic which is based on a simple logical equation. The results from the sex predictions including the best models can be found in output/sex.

The file XModNN_train.py inherits an example training query for XModNN.
The file XModNN_eval.py inherits the evaluation of the trained models.
