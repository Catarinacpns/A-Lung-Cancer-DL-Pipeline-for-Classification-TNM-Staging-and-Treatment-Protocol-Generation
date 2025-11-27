def train_ResNet50_optuna_hyperparams(best_params, train_dataset, val_dataset):
    # Seed
    set_seed(42)
    #with open(file_path, 'r') as f:
        #trial_data = json.load(f)
    
    # Extract hyperparameters
   #best_params = trial_data["hyperparameters"]
    
    # Define where to get the best parameters
    #best_params = best_params_path_SGD_StepLR
    
    # Extract parameters
    learning_rate = best_params['learning_rate']
    batch_size = best_params['batch_size']
    weight_decay = best_params['weight_decay']
    trainable_layers = best_params['trainable_layers']
    dropout_rate = best_params['dropout_rate']
    optimizer_choice = 'SGD'
    scheduler_type = 'StepLR'
    label_smoothing = best_params['label_smoothing']

    # Additional optimizer/scheduler parameters
    momentum = best_params.get('momentum', None)
    step_size = best_params.get('step_size', None)
    gamma = best_params.get('gamma', None)
    factor = best_params.get('factor', None)
    patience = best_params.get('patience', None)
    T_max = best_params.get('T_max', None)
    eta_min = best_params.get('eta_min', None)

    # Dataloader with deterministic seed
    def seed_worker(worker_id):
        np.random.seed(42)
        random.seed(42)

    train_loader = create_dataloader(train_dataset, batch_size, worker_init_fn=seed_worker)
    val_loader = create_dataloader(val_dataset, batch_size, worker_init_fn=seed_worker)

    # Load Pretrained ResNet-50
    model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    num_classes = len(train_dataset.class_map)
    model.fc = nn.Sequential(
        nn.Dropout(dropout_rate),
        nn.Linear(model.fc.in_features, num_classes)
    )
    

    # Freeze layers
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

    # Device setup
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)

    val_f1_scores = []
    best_val_f1 = 0  # Track the best validation F1-score
    epochs_without_improvement = 0  # Counter for epochs without improvement
    patience = 50 
    # Training and validation loop
    for epoch in range(200):  # Run for two epochs to verify consistency
        print(f"\nEpoch {epoch + 1}")

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

        # Calculate validation metrics
        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average='macro'
        )
        val_f1_scores.append(val_f1)
        
        print(f"  Training Loss: {train_loss / len(train_loader.dataset):.4f}", f"  Training Metrics: Accuracy = {train_accuracy:.4f}, Precision = {train_precision:.4f}, "
              f"Recall = {train_recall:.4f}, F1-Score = {train_f1:.4f}")

        print(f"  Validation Loss: {val_loss / len(val_loader.dataset):.4f}", f"  Validation Metrics: Accuracy = {val_accuracy:.4f}, Precision = {val_precision:.4f}, "
              f"Recall = {val_recall:.4f}, F1-Score = {val_f1:.4f}")
        
        # Save metrics to CSV
        with open(metrics_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch + 1,
                train_loss / len(train_loader.dataset), train_accuracy, train_precision,
                train_recall, train_f1, val_loss / len(val_loader.dataset), val_accuracy,
                val_precision, val_recall, val_f1
            ])
            
        # Early stopping logic
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_without_improvement = 0  # Reset counter
        else:
            epochs_without_improvement += 1  # Increment counter

        # Stop trial if no improvement for `patience` epochs
        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}. No improvement for {patience} epochs.")
            break
            
    # Calculate and store classification report properly
    class_report_dict = classification_report(val_labels, val_preds, output_dict=True)
        
    # Save both weights and hyperparameters
    save_data = {
        "model_state_dict": model.state_dict(),
        "hyperparameters": best_params,
        "trainable_layers": trainable_layers,
        "classification_report": class_report_dict
    }
    
    torch.save(save_data, model_save_path)
    print(f"Model and hyperparameters saved to {model_save_path}")