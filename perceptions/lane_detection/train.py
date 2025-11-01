import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from data_loader import LaneDetectionDataset, generate_perceptual_field_data
from model import ConeClassifier

def train_model(train_dataset, val_dataset, model, epochs=250, batch_size=128, learning_rate=0.0015, L = 50, optimizer_ = optim.Adam):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Add weight decay for L2 regularization
    optimizer = optimizer_(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    criterion = nn.CrossEntropyLoss()
    
    best_val_loss = float('inf')
    best_val_accuracy = 0
    epochs_no_improve = 0
    patience = 50  # Reduced patience for small dataset
    min_delta = 0.001  # Minimum change to qualify as improvement
    
    print(f"Starting training with early stopping patience: {patience} epochs")
    print(f"Minimum improvement threshold: {min_delta}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for feat_pair, IoU_pair in train_dataloader:
            features_1, features_2 = torch.unbind(feat_pair, dim=1)
            IoU_1, IoU_2 = torch.unbind(IoU_pair, dim=1)
            batch_size = feat_pair.size(0)
            optimizer.zero_grad()
            
            # Forward pass
            pred_1 = model(features_1)
            pred_2 = model(features_2)
            
            # Calculate loss
            p_gt = F.sigmoid(L * (IoU_1 - IoU_2))
            p_pred = F.sigmoid(pred_1 - pred_2)
            loss = criterion(p_pred, p_gt)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            pred_classification  = pred_1 > pred_2
            gt_classification = IoU_1 > IoU_2
            total += batch_size
            correct += (pred_classification == gt_classification).sum().item()
        
        epoch_loss = running_loss / len(train_dataloader)
        accuracy = 100 * correct / total
        
        # --- Validation Step ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0
        
        with torch.no_grad():
            for feat_pair, IoU_pair in val_dataloader:
                features_1, features_2 = torch.unbind(feat_pair, dim=1)
                IoU_1, IoU_2 = torch.unbind(IoU_pair, dim=1)
                batch_size = feat_pair.size(0)
                
                # Forward pass
                pred_1 = model(features_1)
                pred_2 = model(features_2)
                
                # Calculate loss
                p_gt = F.sigmoid(L * (IoU_1 - IoU_2))
                p_pred = F.sigmoid(pred_1 - pred_2)
                loss = criterion(p_pred, p_gt)
                
                val_running_loss += loss.item()
                
                # Calculate accuracy
                pred_classification  = pred_1 > pred_2
                gt_classification = IoU_1 > IoU_2
                val_total += batch_size
                val_correct += (pred_classification == gt_classification).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        val_loss = val_running_loss / len(val_dataloader)
        
        # Update learning rate based on validation loss
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Log learning rate changes
        if new_lr != old_lr:
            print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Improved early stopping: check both loss and accuracy with minimum delta
        improvement = False
        
        # Check validation loss improvement
        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            improvement = True
            print(f"New best validation loss: {val_loss:.4f}")
        
        # Check validation accuracy improvement
        if val_accuracy > (best_val_accuracy + min_delta):
            best_val_accuracy = val_accuracy
            improvement = True
            print(f"New best validation accuracy: {val_accuracy:.2f}%")
        
        # Save best model if either loss or accuracy improved significantly
        if improvement:
            print(f"Saving best model to best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': val_accuracy,
                'loss': val_loss,
            }, 'best_model.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")
        
        # Early stopping check
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            print(f"Best validation loss: {best_val_loss:.4f}, Best validation accuracy: {best_val_accuracy:.2f}%")
            break

def evaluate_model(model, dataset):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for feat_pair, IoU_pair in dataset:
            features_1, features_2 = torch.unbind(feat_pair, dim=1)
            IoU_1, IoU_2 = torch.unbind(IoU_pair, dim=1)
            batch_size = feat_pair.size(0)

            if not isinstance(features_1, torch.Tensor):
                features_1 = torch.tensor(features_1, dtype=torch.float32)
            if not isinstance(features_2, torch.Tensor):
                features_2 = torch.tensor(features_2, dtype=torch.float32)
            
            # Get predictions
            pred_1 = model(features_1)
            pred_2 = model(features_2)
            
            # Calculate accuracy
            pred_classification  = pred_1 > pred_2
            gt_classification = IoU_1 > IoU_2
            total += batch_size
            correct += (pred_classification == gt_classification).sum().item()
    
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")

def load_model(model_path='model.pth'):
    """Load a pre-trained model from file"""
    model = ConeClassifier()
    
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Handle different save formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'accuracy' in checkpoint:
                print(f"Model loaded from {model_path}")
                print(f"Best validation accuracy: {checkpoint['accuracy']:.2f}%")
                print(f"Best validation loss: {checkpoint['loss']:.4f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"Model loaded successfully from {model_path}")
            
        return model
    except FileNotFoundError:
        print(f"Model file {model_path} not found!")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def main(mode='train', model_path='model.pth'):
    """
    Main function with mode selection
    mode: 'train' to train a new model, 'eval' to only evaluate existing model
    model_path: path to the saved model file
    """
    perceptual_field_data = generate_perceptual_field_data(boundaries, cone_maps)
    
    # --- Create Train/Validation Split ---
    full_dataset = LaneDetectionDataset(perceptual_field_data)
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(0.8 * dataset_size))  # 80% train, 20% validation
    np.random.seed(42)  # for reproducibility
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    # Create train and validation datasets with augmentation for training
    train_dataset = LaneDetectionDataset([(full_dataset.data[i]) for i in train_indices], augment=True)
    val_dataset = LaneDetectionDataset([(full_dataset.data[i]) for i in val_indices], augment=False)  # No augmentation on validation
    
    if mode == 'eval':
        # Only evaluate existing model
        model = load_model(model_path)
        if model is not None:
            print("Evaluating on validation set...")
            evaluate_model(model, val_dataset)
        else:
            print("Cannot evaluate: model loading failed")
    else:
        # Train new model (default behavior)
        model = ConeClassifier()
        train_model(train_dataset, val_dataset, model,
                    epochs = 250, batch_size = 128,
                    learning_rate= 0.0015, L = 50)
        
        # Load the best model and evaluate
        print("Loading best model for final evaluation...")
        best_model = load_model('best_model.pth')
        if best_model:
            print("Final evaluation of best model:")
            evaluate_model(best_model, val_dataset)
        
        # Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'final_config': {
                'input_size': 8,
                'fc1_size': 800,
                'fc2_size': 100,
                'output_size': 1
            }
        }, 'model.pth')

# ============================================================================
# PHASE 4: Pairwise Ranking Training (ADDITIONS)
# ============================================================================

def train_pairwise_model(train_dataset, val_dataset, model, epochs=250, batch_size=128, learning_rate=0.0015):
    """
    Training loop for pairwise ranking approach.

    Uses BCEWithLogitsLoss for binary preference prediction.
    Model learns: given two lane candidates, which one is better?

    KEEP existing train_model() unchanged!
    """
    from data_loader import collate_fn_pairwise

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_pairwise)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_pairwise)

    # Optimizer and scheduler (same as original)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # Binary cross-entropy loss for pairwise ranking
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float('inf')
    best_val_accuracy = 0
    epochs_no_improve = 0
    patience = 50
    min_delta = 0.001

    print(f"Starting pairwise ranking training with early stopping patience: {patience} epochs")
    print(f"Minimum improvement threshold: {min_delta}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for features1, features2, labels in train_dataloader:
            batch_size_curr = features1.size(0)
            optimizer.zero_grad()

            # Compute feature difference
            diff = features1 - features2

            # Forward pass: predict which path is better
            score = model(diff).squeeze()

            # Loss: BCEWithLogits expects raw logits
            loss = criterion(score, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            predictions = (torch.sigmoid(score) > 0.5).float()
            total += batch_size_curr
            correct += (predictions == labels).sum().item()

        epoch_loss = running_loss / len(train_dataloader)
        accuracy = 100 * correct / total

        # --- Validation Step ---
        model.eval()
        val_correct = 0
        val_total = 0
        val_running_loss = 0.0

        with torch.no_grad():
            for features1, features2, labels in val_dataloader:
                batch_size_curr = features1.size(0)

                # Compute difference and predict
                diff = features1 - features2
                score = model(diff).squeeze()

                # Calculate loss
                val_running_loss += criterion(score, labels).item()

                # Calculate accuracy
                predictions = (torch.sigmoid(score) > 0.5).float()
                val_total += batch_size_curr
                val_correct += (predictions == labels).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_loss = val_running_loss / len(val_dataloader)

        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']

        if new_lr != old_lr:
            print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")

        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Early stopping logic
        improvement = False

        if val_loss < (best_val_loss - min_delta):
            best_val_loss = val_loss
            improvement = True
            print(f"New best validation loss: {val_loss:.4f}")

        if val_accuracy > (best_val_accuracy + min_delta):
            best_val_accuracy = val_accuracy
            improvement = True
            print(f"New best validation accuracy: {val_accuracy:.2f}%")

        if improvement:
            print(f"Saving best model to best_model_pairwise.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'accuracy': val_accuracy,
                'loss': val_loss,
            }, 'best_model_pairwise.pth')
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            print(f"Best validation loss: {best_val_loss:.4f}, Best validation accuracy: {best_val_accuracy:.2f}%")
            break

def evaluate_pairwise_model(model, dataset):
    """
    Evaluate pairwise ranking model.

    Metrics:
    - Ranking accuracy: % of correct pairwise preferences
    - Average confidence scores
    """
    from data_loader import collate_fn_pairwise

    model.eval()
    correct = 0
    total = 0
    confidence_scores = []

    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=collate_fn_pairwise)

    with torch.no_grad():
        for features1, features2, labels in dataloader:
            diff = features1 - features2
            score = model(diff).squeeze()

            # Predictions
            predictions = (torch.sigmoid(score) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Confidence
            confidence_scores.extend(torch.sigmoid(score).tolist())

    accuracy = 100 * correct / total
    avg_confidence = np.mean(confidence_scores)

    print(f"Pairwise Ranking Accuracy: {accuracy:.2f}%")
    print(f"Average Confidence: {avg_confidence:.4f}")

def main_pairwise(mode='train', model_path='model_pairwise.pth'):
    """
    Main function for pairwise ranking pipeline.

    KEEP existing main() unchanged!
    """
    from data_loader import generate_pairwise_training_data
    from torch.utils.data import random_split

    # Generate pairwise training data using new pipeline
    print("Generating pairwise training data...")
    full_dataset = generate_pairwise_training_data(boundaries, cone_maps)

    if len(full_dataset) == 0:
        print("Error: No training data generated!")
        return

    # Split train/validation (80/20)
    dataset_size = len(full_dataset)
    train_size = int(0.8 * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Train samples: {train_size}, Validation samples: {val_size}")

    if mode == 'eval':
        # Only evaluate existing model
        model = ConeClassifier()
        try:
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Model loaded from {model_path}")
            evaluate_pairwise_model(model, val_dataset)
        except FileNotFoundError:
            print(f"Model file {model_path} not found!")
    else:
        # Train new model
        model = ConeClassifier()
        train_pairwise_model(train_dataset, val_dataset, model)

        # Load best model and evaluate
        print("\nLoading best model for final evaluation...")
        try:
            best_model = ConeClassifier()
            checkpoint = torch.load('best_model_pairwise.pth', map_location=torch.device('cpu'))
            best_model.load_state_dict(checkpoint['model_state_dict'])
            print("Final evaluation of best model:")
            evaluate_pairwise_model(best_model, val_dataset)
        except FileNotFoundError:
            print("Best model not found, using current model")
            evaluate_pairwise_model(model, val_dataset)

        # Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'final_config': {
                'input_size': 8,
                'fc1_size': 100,
                'output_size': 1,
                'task': 'pairwise_ranking'
            }
        }, model_path)
        print(f"Final model saved to {model_path}")

if __name__ == '__main__':
    import sys

    # Check command line arguments
    if len(sys.argv) > 1:
        # Check for --pairwise flag
        if sys.argv[1] == '--pairwise':
            # Use pairwise ranking pipeline
            mode = sys.argv[2] if len(sys.argv) > 2 else 'train'
            model_path = sys.argv[3] if len(sys.argv) > 3 else 'model_pairwise.pth'
            print("=== Using Pairwise Ranking Pipeline ===")
            main_pairwise(mode, model_path)
        else:
            # Use original pipeline
            mode = sys.argv[1]
            model_path = sys.argv[2] if len(sys.argv) > 2 else 'model.pth'
            print("=== Using Original Point Classification Pipeline ===")
            main(mode, model_path)
    else:
        # Default behavior - use original pipeline
        print("=== Using Original Point Classification Pipeline ===")
        print("Tip: Use '--pairwise' flag to use the new pairwise ranking pipeline")
        print("Example: python train.py --pairwise train")
        main('train')
