from matplotlib import pyplot as plt
from torch import optim, torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from model import ViT, FocalLoss
from pipeline import pipeline_norm_Extension,AugmentedDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


# 1.1 Counting the labels - Eliminate `""" """` to run the test
# Classes/Labels

"""
classes_container:set = set()
debug_dataLoader = DataLoader(data, batch_size=1, shuffle = False)
debug_sample = None

for outputs, labels in debug_dataLoader:
  if hasattr(labels, 'item'):
    classes_container.add(labels.item())
    debug_sample = outputs

plt.imshow(debug_sample[0].permute(1,2,0))
plt.show()
print(classes_container)
"""

data = pipeline_norm_Extension("PreprocessedData")

# 2.2 Create data loaders and necessary containers to store Train and Test Acc
batch_size:int = 64
loss_over_epochs_train:list = []       # Per-epoch average train loss
loss_over_epochs_test:list = []        # Per-epoch average test loss
accuracy_over_epochs_train:list = []   # Per-epoch train accuracy
accuracy_over_epochs_test:list = []    # Per-epoch test accuracy


# 3.2 Split data in `train`, `test` and `validation` sets and create loaders of (b, chn, w, h)
train_data_raw, test_data_raw = train_test_split(data, train_size=0.6, shuffle = False)

train_data_aug = AugmentedDataset(train_data_raw, augment=True)
test_data_aug = AugmentedDataset(test_data_raw, augment=False)


# 1.2 Displaying an augmented image sample - Eliminate `""" """` to run the test
# Augmented samples
"""
img, label = train_data_aug[0]  # Change index to view other samples

# If the image is a single-channel tensor (1, 224, 224), squeeze it
if img.shape[0] == 1:
    img = img.squeeze(0)

# Plot the image
plt.imshow(img.numpy(), cmap='gray')
plt.title(f"Label: {label}")
plt.axis('off')
plt.show()
"""

train_data = DataLoader(train_data_aug,batch_size=batch_size,shuffle=True)
test_data = DataLoader(test_data_aug, batch_size=batch_size,shuffle=False)

# 4 Model set up
device = "cpu" if torch.cuda.is_available else "cpu"
model = ViT(
    patch_size=112,
    emb_dim=384 ,
    heads=8,
    dropout=0.11512622602804817 ,
    n_layers=12  # deeper but still trainable
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=0.00014556891922736168, weight_decay=1e-1)
#scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
weight_tensor = torch.ones(16, dtype=torch.float32).to(device)
criterion = FocalLoss(gamma=2, alpha=weight_tensor, reduction='mean')
            
print("Len train-data {} and len test-data {}".format(len(train_data), len(test_data)))

for epochs in range(30):
    model.train()
    # ----------- TRAINING --------- #
    correct_preds = 0
    total_samples = 0
    train_loss = 0.0

    train_loop = tqdm(train_data, desc=f"Training Epoch {epochs}", leave=False)
    total_batches = len(train_data)
    for batch_index, (inputs, labels) in enumerate(train_loop):
        # print(f"Batch shape: {inputs.shape}, Labels: {labels.shape}")
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        #scheduler.step(epochs + batch_index / total_batches)

        # 5.1 Determning training acc and loss
        train_loss += loss.item() # accumulates loss per batch
        preds = torch.argmax(outputs, dim=1)
        correct_preds += (preds == labels).sum().item()
        total_samples += labels.size(0)


    train_accuracy = correct_preds / total_samples
    train_loss /= total_samples
    accuracy_over_epochs_train.append(train_accuracy) 
    loss_over_epochs_train.append(train_loss)

# Reset the counters before validation
# ---------- VALIDATION ----------- #
    model.eval()
    val_loss = 0.0
    correct_preds = 0
    total_samples = 0

    with torch.no_grad():
        val_loop = tqdm(test_data, desc="Validation {epochs}", leave=False)
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
        val_loss /= total_samples
        accuracy_over_epochs_test.append(val_accuracy)
        loss_over_epochs_test.append(val_loss)

        # Update learning rate if needed
        #scheduler.step(val_accuracy)

    # ---------- LOG ----------
    print(f"Epoch {epochs}: Train Acc = {train_accuracy:.4f} | Test Acc = {val_accuracy:.4f}")


plt.plot(accuracy_over_epochs_train, color='orange',label='Training')
plt.plot(accuracy_over_epochs_test, color = 'red', label = 'Testing')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
torch.save(model.state_dict(), "vit_model_weigh")
