The eXplainable Modular Neural Network called XModNN allows the identification of important
biomarkers in functional hierarchies to classify diseases and clinical parameters in high
throughput sequencing blood expression datasets. Each module and its connections represent
a pathway or gene within the functional hierarchy. This biologically informed architecture
results in a reduction of parameters and an intrinsic feature selection reinforced by the
weighted multi-loss progressive training enabling the successful classification in less
replicates. This workflow in combination with Layer wise Relevance Propagation ensures a
robust post-hoc explanation of the individual module contribution.

Required packages for usage of XModNN in python 3.10.9:
- pandas 2.0.3
- torch 1.13.1+cu117
- numpy 1.23.5
- sklearn 1.3.0
- seaborn 0.12.2

To train, validate and run XModNN you would need ot generate four different files located in /data containing the information of the data (1.), the class-labels per sample id (2.), the structure file to create the ModNN (3.) and the hierarchy file for adjsuting the biological information for the network based on the KEGG or GO or any other functional hierarchy:
Required files in /data:  
1. dataset.csv  
  columnames: sample ids  
  rownames: featurenames  

2. label.csv  
  columnames: sample ids  
  rownames: label  

3. structure.csv  
Based on this file the internal structure of XModNN is created  
F stands for feature, M stands for module, O stands for global network output:  
  F,f1,f2,...  
  M,modulename1,input1,input2,...  
  ...  
  M,modulename10,input101,input102,...  
  O.output_module,input1001,input1002,...  

4. hierarchy.txt  
  This file inherits all individual connections from the functional hierarchy and is used for renaming purposes for the evaluation of XModNN.  
  rowwise: hierarchylevel,modulename,pathwayname(\t)hierarchylevel,modulename,pathwayname(\t)...  

For all required files we included a test dataset in data/logic which is based on a simple logical equation. This includes an artifical dataset, label, structure and hierarchy file to test XModNN. 

The output of XModNN from the sex predictions fomr the publication are included together with the best models in output/sex. No original datasets or proband information can be found there.  

The file XModNN_train.py inherits an example training query for XModNN.  
The file XModNN_eval.py inherits the evaluation of the trained models.  
