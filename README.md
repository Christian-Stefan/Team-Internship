# Team-Internship
## Model Variations (by Lan Ma)
4 modified versions were tested to evaluate performance:
1. *Local-CNN*  
   Only the `local branch` replaced with a 3-layer CNN.  
   other two branches unchanged.
   performance:  Accuracy: 0.6743  F1 Score: 0.6608
   File: `model_localCNN3.ipynb`

2. *Local-ResNet*
   Only the `local branch` replaced with a ResNet structure.  
   other two branches unchanged.
   performance:  Accuracy: 0.7798  F1 Score: 0.7609
   File: `model_localResNet.ipynb`

3. *Context-ResNet* 
   Only the `context branch` replaced with a ResNet structure.  
   other two branches unchanged.
   performance: Accuracy: 0.6743 F1 Score: 0.6608 
   File: `model_contextResNet.ipynb`

4. *Full-ResNet*
   Both `local branch` and `context branch` replaced with ResNet.  
   `radiomics` branch unchanged.
   performance can't be calculated well due to insufficient computational power of my latop.
   File: `model_ResNet2.ipynb`
