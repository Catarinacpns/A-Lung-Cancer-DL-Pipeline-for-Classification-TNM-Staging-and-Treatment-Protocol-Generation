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


def objective(trial):
    # Set seed for trial reproducibility
    set_seed(42)
    gc.collect()
    
    # Hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)  # Smaller range for stability
    batch_size = trial.suggest_categorical('batch_size', [16, 32])
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)  # Narrow range for weight decay
    trainable_layers = trial.suggest_int('trainable_layers', 2, 4, step=1)
    dropout_rate = trial.suggest_float('dropout_rate', 0.15, 0.3)
    optimizer_choice = trial.suggest_categorical('optimizer', ['Adam', 'SGD', 'RMSprop'])
    scheduler_type = trial.suggest_categorical('scheduler', ['StepLR', 'CosineAnnealingLR', 'ReduceLROnPlateau'])
    label_smoothing = trial.suggest_float('label_smoothing', 0.0, 0.15)

    # Additional optimizer/scheduler parameters
    momentum = trial.suggest_float('momentum', 0.7, 0.99) if optimizer_choice == 'SGD' else None
    step_size = trial.suggest_int('step_size', 1, 6) if scheduler_type == 'StepLR' else None
    gamma = trial.suggest_float('gamma', 0.5, 0.9) if scheduler_type == 'StepLR' else None
    factor = trial.suggest_float('factor', 0.2, 0.5) if scheduler_type == 'ReduceLROnPlateau' else None
    patience = trial.suggest_int('patience', 3, 5) if scheduler_type == 'ReduceLROnPlateau' else None
    T_max = trial.suggest_int('T_max', 5, 9) if scheduler_type == 'CosineAnnealingLR' else None
    eta_min = trial.suggest_float('eta_min', 1e-6, 1e-4) if scheduler_type == 'CosineAnnealingLR' else None

    # Print trial configuration
    print(f"\nStarting Trial {trial.number + 1}")
    print(f"Hyperparameters:")
    print(f"  Learning Rate      = {learning_rate:.6e}")
    print(f"  Batch Size         = {batch_size}")
    print(f"  Weight Decay       = {weight_decay:.6e}")
    print(f"  Trainable Layers   = {trainable_layers}")
    print(f"  Dropout Rate       = {dropout_rate:.2f}")
    print(f"  Optimizer          = {optimizer_choice}")
    if momentum:
        print(f"    Momentum         = {momentum:.2f}")
    print(f"  Scheduler          = {scheduler_type}")
    if scheduler_type == 'StepLR':
        print(f"    Step Size        = {step_size}")
        print(f"    Gamma            = {gamma:.2f}")
    elif scheduler_type == 'ReduceLROnPlateau':
        print(f"    Factor           = {factor:.2f}")
        print(f"    Patience         = {patience}")
    elif scheduler_type == 'CosineAnnealingLR':
        print(f"    T_max            = {T_max}")
        print(f"    Eta Min          = {eta_min:.6e}")
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
    if optimizer_choice == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_choice == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    elif optimizer_choice == 'RMSprop':
        optimizer = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=learning_rate, weight_decay=weight_decay)
    # Scheduler
    if scheduler_type == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience)
    elif scheduler_type == 'CosineAnnealingLR':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

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
                "type": optimizer_choice,
                "parameters": {"momentum": momentum}
            },
            "scheduler": {
                "type": scheduler_type,
                "parameters": {
                    "step_size": step_size,
                    "gamma": gamma,
                    "factor": factor,
                    "patience": patience,
                    "T_max": T_max,
                    "eta_min": eta_min,
                }
            },
            "label_smoothing": label_smoothing
        },
        "train_metrics": [],
        "val_metrics": []
    }

    val_f1_scores = []
    for epoch in range(10):
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
        
        print(f"  Training Loss: {train_loss / len(train_loader.dataset):.4f}", f"  Training Metrics: Accuracy = {train_accuracy:.4f}, Precision = {train_precision:.4f}, "
              f"Recall = {train_recall:.4f}, F1-Score = {train_f1:.4f}")
            
        print(f"  Validation Loss: {val_loss / len(val_loader.dataset):.4f}", f"  Validation Metrics: Accuracy = {val_accuracy:.4f}, Precision = {val_precision:.4f}, "
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
