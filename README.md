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




