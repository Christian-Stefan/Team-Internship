# Local Context model branch

This is a new temporary branch for the results of different local branch configurations


## Residual Layer with attention

This method has given a score of:

- 20 epochs:
Accuracy: 0.7064
F1 Score: 0.6888

- 40 epochs:
Accuracy: 0.8211
F1 Score: 0.8121


- 60 epochs:
Accuracy: 0.8716
F1 Score: 0.8726

However this score was with incorrect test data, so the accuracies are not correct.

This latest version is correctly tested. But this time a top 3 accuracy is shown, this means top-1 is the accuracy for the best prediction to be accurate, top 2 the accuracy for the best prediction or the second prediction to be accurate etc.

this gave for the best model so far for this branch (which is the Local context residualLayer alt notebook):

- 35 epochs:  
Top-1 Accuracy: 0.5655  
Top-1 F1 Score:  0.5612  
Top-2 Accuracy: 0.7379  
Top-3 Accuracy: 0.8759  

Some outputs for this:  

True: Squamous Cell Carcinoma, Top-1: Squamous Cell Carcinoma (46.6%), Top-2: Adenoid Cystic Carcinoma (34.2%), Top-3: Small Cell Lung Cancer (SCLC) (18.2%)  

True: Granuloma, Top-1: Granuloma (98.3%), Top-2: Carcinoid Tumors (1.5%), Top-3: Sarcoidosis (0.1%)  

True: Large Cell (Undifferentiated) Carcinoma, Top-1: Squamous Cell Carcinoma (63.4%), Top-2: Large Cell (Undifferentiated) Carcinoma (20.3%), Top-3: Lymphoma (10.8%)  

True: Bronchioloalveolar Hyperplasia, Top-1: Sarcomatoid Carcinoma (44.8%), Top-2: Adenocarcinoma (16.8%), Top-3: Bronchioloalveolar Hyperplasia (16.1%)  

True: Squamous Cell Carcinoma, Top-1: Lymphoma (54.2%), Top-2: Bronchioloalveolar Hyperplasia (16.6%), Top-3: Squamous Cell Carcinoma (15.7%)  


as can be seen in the examples, it really helps for this classification to not focus on the highest likelihood only. If you were to do this you would assume this model is bad: 57% accuracy. However as can be seen in the outputs, some nodules are quite similar to each other. Like the different types of carcinoma. Which really brings the accuracy of top-1 accuracy down. But most of the time the other versions of carcinoma are also in the top-3, which increases the accuracy a lot: 88%.  

Local context residualLayer altv2 gives this:

- 40 epochs:
Top-1 Accuracy: 0.6552
Top-1 F1 Score:  0.6590
Top-2 Accuracy: 0.8276
Top-3 Accuracy: 0.8828

which while having a better top-1 accuracy, does not have a significant better accuracy in the top-3 accuracy.  

The pipeline which gives the evaluation output for 1 scan is explained next:  

Model_data_preparation.py:  

contains a class ModelDataInput, which takes for 1 scan:  
- the json path of the nodule  
- the scan path of the preprocessed nodule  
- whether the input is the local dicom scans (nodule only) or the context dicom scans (lung area)  
- the naming method of the dicom files  

It outputs for 1 scan:  
- the local volume or the context volume  
- the radiomics  
- the label  

Model_classification.py:  

contains a class ModelClassification, which takes for 1 scan:  
- the modelpkl (for now only works with trained models from the Local_context_residualLayer alt.ipynb notebook)  
- the local volume  
- the context volume  
- the radiomics  
- the label  

It outputs for 1 scan:  
The true label and the Top 3 prediction



