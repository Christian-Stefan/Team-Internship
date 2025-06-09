from training import k_fold_Cross_Validation
from pipeline import pipeline_norm_Extension
from PSO import param_grid_generator


# 1. Declaring variables
history:list = []
ViT_history:list = []
best_score = 999999999999

# 2. Preprocessing data consisting of conversion-related operations (e.g. turn out the data at hand in tensor objects of type {img, label})
data = pipeline_norm_Extension("PreprocessedData")

print(f"Data length: {len(data)}")
# 3. Hyperparameter optimization - 50 variations of ViT
max_variations:int = 50
for variation in range(max_variations):
    param_grid = param_grid_generator() # 3.1. Generate a new variation consisting of a set of hyperparameters
    model_optimization = k_fold_Cross_Validation(folds=5) 
    print(f"The grid of hyperparameters consists of...{param_grid}")
    score, acc, val_loss, highest_accuracy, history_Train_Loss, history_Validation_Loss = model_optimization.training_and_val(data=data,
                                                                                                                              train_size=0.80,
                                                                                                                              param_grid=param_grid,
                                                                                                                              model_identifier=variation) # 3.2. Undergoing training and validation having equiped param_grid

    # 4. Keeping count of best scores and print out results
    if score < best_score:
                print('Found new best hyperparameters: '
                    'learning rate ', param_grid['learning_rate'],
                    'patch_size', param_grid['patch_size'],
                    'emb_size', param_grid['emb_size'],
                    'dropout', param_grid['dropout'],
                    'n_heads', param_grid['n_heads'],
                    'score ', score,
                    '\n')
                best_score = score

    else:
        print('Found NO best hyperparameters: '
                    'learning rate ', param_grid['learning_rate'],
                    'patch_size', param_grid['patch_size'],
                    'emb_size', param_grid['emb_size'],
                    'dropout', param_grid['dropout'],
                    'n_heads', param_grid['n_heads'],
                    'score ', score,
                    '\n')
        
    history.append([score,param_grid,acc,val_loss,highest_accuracy])
    # Save the history_Learning in a CSV file
    ViT_history.append(history_Train_Loss)
    ViT_history.append(history_Validation_Loss)