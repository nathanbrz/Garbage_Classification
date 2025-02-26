import torch
import torch.nn as nn
import torch.optim as optim
from transformers import DistilBertTokenizer
from transformers import AdamW
import numpy as np

# Import loaders from dataset_loader.py
from dataset_loader import train_loader, val_loader, test_loader
# Import model from model_architecture.py
from model_architecture import MultiModalGarbageClassifier

def train_one_epoch(model, loader, tokenizer, optimizer, criterion, device):
    """
    Train the model for one epoch.

    Parameters:
        model (nn.Module): Multi-modal model.
        loader (DataLoader): Training data loader.
        tokenizer (DistilBertTokenizer): DistilBERT tokenizer.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        criterion (nn.Module): Loss function.
        device (str): Device to run the model on.

    Return:
        epoch_loss (float): Average loss of the epoch.
        epoch_acc (float): Average accuracy of the epoch.
    """

    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    for images, text_descriptions, labels in loader:
        # Move data to device
        images = images.to(device)
        labels = labels.to(device)

        # Tokenize text descriptions
        encoding = tokenizer (
            list(text_descriptions),
            padding= "longest", # Pad to the longest text description
            truncation= True, # Truncate if the text exceeds the max_length
            return_tensors= "pt"
        )

        # Move data to device
        encoding = {key: val.to(device) for key, val in encoding.items()}

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images, encoding)
        loss = criterion(outputs, labels)

        # Compute predictions
        _, preds = torch.max(outputs, 1)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += images.size(0)
    
    # Compute epoch loss and accuracy
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples

    return epoch_loss, epoch_acc

def evaluate(model, loader, tokenizer, criterion, device):
    """
    Evaluate the model on the validation or test set.
    
    Parameters:
        model (nn.Module): Multi-modal model.
        loader (DataLoader): Validation or test data loader.
        tokenizer (DistilBertTokenizer): DistilBERT tokenizer.
        criterion (nn.Module): Loss function.
        device (str): Device to run the model on.
    
    Return:
        epoch_loss (float): Average loss of the epoch.
        epoch_acc (float): Average accuracy of the epoch.
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    with torch.no_grad():
        for images, text_descriptions, labels in loader:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device)

            # Tokenize text descriptions
            encoding = tokenizer (
                list(text_descriptions),
                padding= "longest",
                truncation= True,
                return_tensors= "pt"
            )

            # Move data to device
            encoding = {key: val.to(device) for key, val in encoding.items()}

            # Forward pass
            outputs = model(images, encoding)
            loss = criterion(outputs, labels)

            # Compute predictions
            _, preds = torch.max(outputs, 1)

            # Statistics
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += images.size(0)
        
    # Compute epoch loss and accuracy
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples

    return epoch_loss, epoch_acc.item()

def train_model(
        model,
        train_loader,
        val_loader,
        tokenizer,
        criterion,
        optimizer,
        device,
        num_epochs= 5
):
    """
    Full training loop. Saves the best model weights based on validation accuracy.

    Parameters:
        model (nn.Module): Multi-modal model.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        tokenizer (DistilBertTokenizer): DistilBERT tokenizer.
        criterion (nn.Module): Loss function.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (str): Device to run the model on.
        num_epochs (int): Number of epochs to train.

    Return:
        model (nn.Module): Best model based on validation accuracy.
    """

    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print("-" * 10)

        # Training
        train_loss, train_acc = train_one_epoch(model, train_loader, tokenizer, optimizer, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        # Validation
        val_loss, val_acc = evaluate(model, val_loader, tokenizer, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")
            print("Best model saved!")
        
    print(f"Training Complete! Best Validation Accuracy: {best_acc:.4f}")
    return model

if __name__ == "__main__":
    # 1. Setup device
    device = torch.device("cuda")
    print("Using device:", device)

    # 2. Initialize model
    model = MultiModalGarbageClassifier(num_classes=4).to(device)

    # 3. Initialize DistilBERT tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # 4. Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-5)

    # 5. Train the model
    model = train_model(
        model,
        train_loader,
        val_loader,
        tokenizer,
        criterion,
        optimizer,
        device,
        num_epochs=8
    )

    # 6. Evaluate the model on the test set
    print("Evaluating the model on the test set...")
    test_loss, test_acc = evaluate(model, test_loader, tokenizer, criterion, device)
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
