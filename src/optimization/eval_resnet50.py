def evaluate_resnet50(checkpoint_path, test_data_path, batch_size=32, save_path= 'test_metrics.csv'):
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # ResNet50 expects 224x224 input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize to ImageNet's scale
    ])

    # Load test dataset
    test_files = glob.glob(os.path.join(test_data_path, "*"))
    test_dataset = LungCancerDataset(test_files, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_classes = len(test_dataset.class_map)
    
    model.fc = nn.Sequential(
        nn.Dropout(0.5),  # Match training dropout rate
        nn.Linear(model.fc.in_features, num_classes)
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Evaluation
    test_loss = 0
    test_preds, test_labels = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating Test Set", unit="batch"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)

            _, preds = torch.max(outputs, 1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    # Compute test metrics
    test_loss /= len(test_loader.dataset)
    test_accuracy = accuracy_score(test_labels, test_preds)
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        test_labels, test_preds, average='weighted'
    )

    # Print results
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")

    # Classification Report
    class_names = list(test_dataset.class_map.keys())
    print("\nClassification Report:\n")
    print(classification_report(test_labels, test_preds, target_names=class_names))

    # Save test metrics
    test_metrics_file = os.path.join(os.path.dirname(checkpoint_path), save_path)
    with open(test_metrics_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Test Loss", "Test Accuracy", "Test Precision", "Test Recall", "Test F1"])
        writer.writerow([test_loss, test_accuracy, test_precision, test_recall, test_f1])

    print(f"Test metrics saved to {test_metrics_file}")
    
    print("\nConfusion Matrix:\n")
    print(confusion_matrix(test_labels, test_preds))