# Team-Internship
Model Variations (by Lan Ma.20250528)

Original version (by Vani): Accuracy: 0.6422 F1 Score: 0.6151

After spliting dataset into 3 parts(80:10:10), 4 modified versions were tested to evaluate performance:

1.Local-CNN

- Only the local branch replaced with a 3-layer CNN.
- Other two branches unchanged.
- Training Epochs: 80 (differs from other 3 models of 20 epochs)
- Performance: Accuracy: 0.5616 F1 Score: 0.5579
- File: model_localCNN3.ipynb

2.Local-ResNet

- Only the local branch replaced with a ResNet structure.
- Other two branches unchanged.
- Training Epochs: 20
- Performance: Accuracy: 0.6096 F1 Score: 0.5887
- File: model_localResNet.ipynb

3.Context-ResNet

- Only the context branch replaced with a ResNet structure.
- Pther two branches unchanged.
- Training Epochs: 20
- Performance: Accuracy: 0.5753 F1 Score: 0.5561
- File: model_contextResNet.ipynb

4.Full-ResNet

- Both local branch and context branch replaced with ResNet, add Attention Mechanism.
- radiomics branch unchanged.
- Training Epochs: 20
- Performance: Accuracy: 0.6575 F1 Score: 0.6594
- File: model_ResNet2.ipynb

------------------------------------------------------------------------------------------------------
## Discarded version

Due to the data leakage, the models perform exceptionally well as followed,

1.Local-CNN

- Only the local branch replaced with a 3-layer CNN.
- Other two branches unchanged.
- Training Epochs: 40
- Performance: Accuracy: 0.9128 F1 Score: 0.9128
- File: model_localCNN3.ipynb

2.Local-ResNet

- Only the local branch replaced with a ResNet structure.
- Other two branches unchanged.
- Training Epochs: 20
- Performance: Accuracy: 0.7982 F1 Score: 0.7692
- File: model_localResNet.ipynb

3.Context-ResNet

- Only the context branch replaced with a ResNet structure.
- Pther two branches unchanged.
- Training Epochs: 20
- Performance: Accuracy: 0.6101 F1 Score: 0.5776
- File: model_contextResNet.ipynb

4.Full-ResNet

- Both local branch and context branch replaced with ResNet.
- radiomics branch unchanged.
- Training Epochs: 20
- Performance: Accuracy: 0.9266 F1 Score: 0.9256
- File: model_ResNet2.ipynb
