# Team-Internship

## Model Variations (by Lan Ma)
Original version (by Vani): Accuracy: 0.6422 F1 Score: 0.6151

4 modified versions were tested to evaluate performance:
1. *Local-CNN*  
   - Only the `local branch` replaced with a 3-layer CNN.  
   - Other two branches unchanged.
   - Performance:  Accuracy: 0.6743  F1 Score: 0.6608
   - File: `model_localCNN3.ipynb`

2. *Local-ResNet*
   - Only the `local branch` replaced with a ResNet structure.  
   - Other two branches unchanged.
   - Performance:  Accuracy: 0.7798  F1 Score: 0.7609
   - File: `model_localResNet.ipynb`

3. *Context-ResNet* 
   - Only the `context branch` replaced with a ResNet structure.  
   - Pther two branches unchanged.
   - Performance: Accuracy: 0.6743 F1 Score: 0.6608 (same as the ones of *Local-CNN*, i don't think it has beed trained successfully, additionally it took over 10h through 20 epochs)
   - File: `model_contextResNet.ipynb`

4. *Full-ResNet*
   - Both `local branch` and `context branch` replaced with ResNet.  
   - `radiomics branch` unchanged.
   - Performance can't be calculated well due to insufficient computational power of my latop.
   - File: `model_ResNet2.ipynb`
