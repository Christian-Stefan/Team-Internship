from sklearn.model_selection import KFold, train_test_split
import torch
from torch.utils.data import DataLoader
from model import ViT, FocalLoss
from tqdm import tqdm
from torch import nn, optim
import numpy as np
import cuda  


class k_fold_Cross_Validation:
    def __init__(self, folds:int, visualisation:bool = False):
        self.fold = folds
        self.VIS = visualisation

    def training_and_val(self,
                         data, 
                         train_size:float,
                         param_grid,
                         model_identifier:int): # Model's unique identifier
        
        # Create data loaders and necessary containers to store Train and Test Acc:
        loss_over_folds_train:list = []       # Per-epoch average train loss
        loss_over_folds_test:list = []        # Per-epoch average test loss
        acc_over_folds_test:list = []   # Per-epoch train accuracy
        acc_over_folds_train:list = []    # Per-epoch test accuracy

        HighestAccuracy:int = 0 # Necessary to compute CV score (e.g., a sort of MAX to compare against)
        batch_size:int =  5
        
        # `Unpacking` hyperparameters from param_grid
  
        # Data split
        data_temp, data_test = train_test_split(data, train_size=train_size, shuffle=True)
        # -------------------- INIT K_FOLD and training criterias --------------------- #
        k_fold = KFold(self.fold, shuffle=True)
        
        print(f"Raw data holds {len(data_temp)} samples, with a batch_size of {batch_size}, each containing...")

        for fold_counter ,(train_idx, val_idx) in enumerate(k_fold.split(data_temp)):
            train_set = torch.utils.data.dataset.Subset(data_temp, train_idx)
            val_set = torch.utils.data.dataset.Subset(data_temp, val_idx)
            trainLoader = DataLoader(train_set, batch_size, True)
            valLoader = DataLoader(val_set, batch_size, False)
            
            print(f"Fold: {fold_counter} -- > Train set of length {len(trainLoader)}\nTest set of length {len(valLoader)}")

            # `Init` ViT and inject hyperparameters
            device = "cpu" if torch.cuda.is_available else "cpu"
            model = ViT(patch_size=param_grid['patch_size'],
                        emb_dim=param_grid['emb_size'],
                        heads=param_grid['n_heads'],
                        dropout=param_grid['dropout']).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=param_grid['learning_rate'])
            
            weight_tensor = torch.ones(17, dtype=torch.float32).to(device)
            criterion = FocalLoss(gamma=2, alpha=weight_tensor, reduction='mean')
            
            
            temp_acc_train:list= [] # Temporary Containers
            temp_acc_val:list= []
            CVScore:int = 0 
  
            model.train()
            for epochs in range(10):
                # ----------- TRAINING --------- #
                correct_preds = 0
                total_samples = 0
                train_loss = 0.0

                train_loop = tqdm(trainLoader, desc=f"Training Epoch {epochs}", leave=False)
                for inputs, labels in train_loop:
                    # print(f"Batch shape: {inputs.shape}, Labels: {labels.shape}")
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
 
                    # 5.1 Determning training acc and loss
                    train_loss += loss.item() # accumulates loss per batch
                    preds = torch.argmax(outputs, dim=1)
                    correct_preds += (preds == labels).sum().item()
                    total_samples += labels.size(0)


                train_accuracy = correct_preds / total_samples
                temp_acc_train.append(train_accuracy) 
                train_loss /= total_samples

            # Reset the counters before validation
            # ---------- VALIDATION ----------- #
            model.eval()
            val_loss = 0.0
            correct_preds = 0
            total_samples = 0

            with torch.no_grad():
                val_loop = tqdm(valLoader, desc="Validation", leave=False)
                for inputs, labels in val_loop:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs=model(inputs)
                    loss = criterion(outputs, labels)
        
                    # 5.2 Determining validation  and loss
                    val_loss += loss.item() # accumulates loss per batch
                    preds = torch.argmax(outputs, dim=1)
                    correct_preds += (preds==labels).sum().item()
                    total_samples += labels.size(0)

                val_accuracy = correct_preds / total_samples
                temp_acc_val.append(val_accuracy)
                val_loss /= total_samples


            print(f"Fold {fold_counter} ends having acquired AVG train acc: {val_accuracy:.6f}, AVG test acc: {train_accuracy:.6f}") 
            acc_over_folds_test.append(np.mean(temp_acc_train))
            acc_over_folds_train.append(np.mean(temp_acc_val))
            loss_over_folds_train.append(train_loss) # Appending the train and val loss per fold
            loss_over_folds_test.append(val_loss)

             # Track the highest accuracy
            if val_accuracy > HighestAccuracy:
                HighestAccuracy = val_accuracy

            # Calculate CV Score once per fold
            CVScore+=val_loss
            
            print('Temporary CVscore ', CVScore)
            print('Go to next fold')
            
        print(f"-----------> Cross validation complete for model with number {model_identifier}")
        # Save weights
        torch.save(model.state_dict(),f'Backlog\\WeightsOfTheModel{model_identifier}.pth') 
        totalCVscore = CVScore/self.fold
        print("CV score: ", totalCVscore)

        return totalCVscore, np.mean(acc_over_folds_test), np.mean(loss_over_folds_test), HighestAccuracy, acc_over_folds_train, loss_over_folds_train
            


  