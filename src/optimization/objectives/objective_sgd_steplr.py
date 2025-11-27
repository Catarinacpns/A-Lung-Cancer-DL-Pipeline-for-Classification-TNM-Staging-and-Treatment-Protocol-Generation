import gc
import json
import os
import numpy as np
import random
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from training.reproducibility import set_seed
from dataloading.dataloader import create_dataloader

# Optuna Objective Function
def opbjective_sgd_steplr(trial):
    # Set seed for trial reproducibility
    set_seed(42)
    gc.collect()
    
    # Hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-7, 1e-5, log=True)  # Smaller range for stability
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    weight_decay = trial.suggest_float('weight_decay', 1e-4, 1e-2, log=True)  # Narrow range for weight decay
    trainable_layers = trial.suggest_int('trainable_layers', 2, 4, step=2)
    dropout_rate = trial.suggest_float('dropout_rate', 0.16, 0.30)
    scheduler_type = 'StepLR'
    label_smoothing = trial.suggest_float('label_smoothing', 0.05, 0.20)

    # Additional optimizer/scheduler parameters
    momentum = trial.suggest_float('momentum', 0.79, 0.99)
    step_size = trial.suggest_int('step_size', 3, 5) if scheduler_type == 'StepLR' else None
    gamma = trial.suggest_float('gamma', 0.5, 0.9) if scheduler_type == 'StepLR' else None


    # Print trial configuration
    print(f"\nStarting Trial {trial.number + 1}")
    print(f"Hyperparameters:")
    print(f"  Learning Rate      = {learning_rate:.6e}")
    print(f"  Batch Size         = {batch_size}")
    print(f"  Weight Decay       = {weight_decay:.6e}")
    print(f"  Trainable Layers   = {trainable_layers}")
    print(f"  Dropout Rate       = {dropout_rate:.2f}")
    print(f"  Optimizer          = SGD")
    if momentum:
        print(f"    Momentum         = {momentum:.2f}")
    print(f"  Scheduler          = {scheduler_type}")
    print(f"    Step Size        = {step_size}")
    print(f"    Gamma            = {gamma:.2f}")

    print(f"  Label Smoothing    = {label_smoothing:.4f}\n")
    
    def seed_worker(worker_id):
        np.random.seed(42)
        random.seed(42)

    # Dataloader
    train_loader = create_dataloader(train_dataset, batch_size, worker_init_fn=seed_worker)
    val_loader = create_dataloader(val_dataset, batch_size, worker_init_fn=seed_worker)

    # Load Pretrained ResNet-50
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_classes = len(train_dataset.class_map)
    #model = model.to(torch.float32)

    # Replace FC Layer
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(model.fc.in_features, num_classes)
    )
    
    # Freeze Layers
    for param in model.parameters():
        param.requires_grad = False
    for layer in list(model.children())[-trainable_layers:]:
        for param in layer.parameters():
            param.requires_grad = True


    # Optimizer
    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Loss Function
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Set Device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Training and Validation
    trial_results = {
        "hyperparameters": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "weight_decay": weight_decay,
            "trainable_layers": trainable_layers,
            "dropout_rate": dropout_rate,
            "optimizer": {
                "type": 'SGD',
                "parameters": {"momentum": momentum}
            },
            "scheduler": {
                "type": scheduler_type,
                "parameters": {
                    "step_size": step_size,
                    "gamma": gamma
                }
            },
            "label_smoothing": label_smoothing
        },
        "train_metrics": [],
        "val_metrics": []
    }

    val_f1_scores = []
    best_val_f1 = 0  # Track the best validation F1-score
    epochs_without_improvement = 0  # Counter for epochs without improvement
    patience = 8 
    early_stop_threshold = 0.37 
    for epoch in range(15):
        # Training
        model.train()
        train_loss = 0
        train_preds, train_labels = [], []
        with tqdm(train_loader, desc=f"Training Epoch {epoch + 1}", unit="batch") as tbar:
            for inputs, labels in tbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * inputs.size(0)

                _, preds = torch.max(outputs, 1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

                tbar.set_postfix(loss=loss.item())
                
                del inputs, labels, outputs, loss
                gc.collect()

        # Calculate training metrics
        train_accuracy = accuracy_score(train_labels, train_preds)
        train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
            train_labels, train_preds, average='macro'
        )

        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_labels = [], []
        with tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}", unit="batch") as tbar:
            with torch.no_grad():
                for inputs, labels in tbar:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)

                    _, preds = torch.max(outputs, 1)
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())

                    tbar.set_postfix(loss=loss.item())
                    del inputs, labels, outputs, loss
                    gc.collect()

        # Calculate validation metrics
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='macro')
        val_f1_scores.append(val_f1)
        
        print(f"Training Loss: {train_loss / len(train_loader.dataset):.4f}", f"Training Metrics: Accuracy = {train_accuracy:.4f}, Precision = {train_precision:.4f}, "
              f"Recall = {train_recall:.4f}, F1-Score = {train_f1:.4f}")
            
        print(f"Validation Loss: {val_loss / len(val_loader.dataset):.4f}", f"Validation Metrics: Accuracy = {val_accuracy:.4f}, Precision = {val_precision:.4f}, "
              f"Recall = {val_recall:.4f}, F1-Score = {val_f1:.4f}")
        

        # Save epoch metrics
        trial_results["train_metrics"].append({
            "epoch": epoch + 1,
            "loss": train_loss / len(train_loader.dataset),
            "accuracy": train_accuracy,
            "precision": train_precision,
            "recall": train_recall,
            "f1_score": train_f1   
        })
            
        trial_results["val_metrics"].append({
            "epoch": epoch + 1,
            "loss": val_loss / len(val_loader.dataset),
            "accuracy": val_accuracy,
            "precision": val_precision,
            "recall": val_recall,
            "f1_score": val_f1})
        
        # Early stopping logic
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0  # Reset counter
        else:
            epochs_without_improvement += 1  # Increment counter

        # Early stopping logic for low mean F1-score in first 4 epochs
        if epoch == 3:  # After 4 epochs (0-indexed)
            mean_f1_first_4 = sum(val_f1_scores) / 4
            if mean_f1_first_4 < early_stop_threshold:
                print(f"Early stopping triggered at epoch {epoch + 1}. Mean F1-score for first 4 epochs ({mean_f1_first_4:.4f}) is below {early_stop_threshold}.")
                break

        # Stop trial if no improvement for `patience` epochs
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}. No improvement for {patience} epochs.")
            break
            
    # Calculate and store classification report properly
    class_report_dict = classification_report(val_labels, val_preds, output_dict=True)
    trial_results["classification_report"] = class_report_dict  
    
    trial_results["final_metrics"] ={
        "val_loss": val_loss / len(val_loader.dataset),
        "val_accuracy": val_accuracy,
        "val_precision": val_precision,
        "val_recall": val_recall,
        "val_f1": val_f1}

    # Save trial results
    trial_file = os.path.join(RESULTS_DIR, f"trial_{trial.number + 1}.json")
    with open(trial_file, 'w') as f:
        json.dump(trial_results, f, indent=4)
        
    gc.collect()

    mean_val_f1 = sum(val_f1_scores) / len(val_f1_scores)
    return mean_val_f1