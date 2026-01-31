import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from perceptions.lane_detection.data_loader import generate_all_perceptual_field_data
from perceptions.lane_detection.dataset import LaneDetectionDataset
from perceptions.lane_detection.model import ConeClassifier


def train_model(
    train_dataset,
    val_dataset,
    model,
    epochs=250,
    batch_size=32,
    learning_rate=0.001,
    L=50,
    optimizer_=optim.AdamW,
    eta_min=1e-6,
    patience=20,
):
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Add weight decay for L2 regularization
    # Add weight decay for L2 regularization
    optimizer = optimizer_(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer, mode="min", factor=0.3, patience=10
    )
    # Binary cross entropy loss for pairwise ranking
    # We're predicting probability that candidate 1 is better than candidate 2
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    best_val_accuracy = 0
    epochs_no_improve = 0
    min_delta = 0.001  # Minimum change to qualify as improvement

    # Track metrics for plotting
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    epoch_numbers = []

    print(f"Starting training with early stopping patience: {patience} epochs")
    print(f"Minimum improvement threshold: {min_delta}")

    # Debug: Check class distribution
    train_pos = sum(
        1 for _, iou_pair in train_dataset.data if iou_pair[0] > iou_pair[1]
    )
    train_total = len(train_dataset)
    val_pos = sum(1 for _, iou_pair in val_dataset.data if iou_pair[0] > iou_pair[1])
    val_total = len(val_dataset)
    print(f"\n=== Class Distribution Debug ===")
    print(
        f"Train: {train_pos}/{train_total} ({100*train_pos/train_total:.1f}%) have IoU1 > IoU2"
    )
    print(
        f"Val:   {val_pos}/{val_total} ({100*val_pos/val_total:.1f}%) have IoU1 > IoU2"
    )
    print(f"================================\n")

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
            pred_1 = model(features_1).squeeze()  # (batch_size, 1) -> (batch_size,)
            pred_2 = model(features_2).squeeze()  # (batch_size, 1) -> (batch_size,)

            # Calculate loss
            p_gt = F.sigmoid(L * (IoU_1 - IoU_2))
            p_pred = F.sigmoid(pred_1 - pred_2)
            loss = criterion(p_pred, p_gt)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            pred_classification = pred_1 > pred_2
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
                pred_1 = model(features_1).squeeze()  # (batch_size, 1) -> (batch_size,)
                pred_2 = model(features_2).squeeze()  # (batch_size, 1) -> (batch_size,)

                # Calculate loss
                p_gt = F.sigmoid(L * (IoU_1 - IoU_2))
                p_pred = F.sigmoid(pred_1 - pred_2)
                loss = criterion(p_pred, p_gt)

                val_running_loss += loss.item()

                # Calculate accuracy
                pred_classification = pred_1 > pred_2
                gt_classification = IoU_1 > IoU_2
                val_total += batch_size
                val_correct += (pred_classification == gt_classification).sum().item()

        val_accuracy = 100 * val_correct / val_total
        val_loss = val_running_loss / len(val_dataloader)

        # Update learning rate based on validation loss
        old_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        # Log learning rate changes
        # if new_lr != old_lr:
        #     print(f"Learning rate reduced from {old_lr:.6f} to {new_lr:.6f}")

        print(
            f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Track metrics for plotting
        train_losses.append(epoch_loss)
        train_accuracies.append(accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        epoch_numbers.append(epoch + 1)

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
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "accuracy": val_accuracy,
                    "loss": val_loss,
                },
                "best_model.pth",
            )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs")

        # Early stopping check
        if epochs_no_improve >= patience:
            print(
                f"Early stopping triggered after {patience} epochs with no improvement."
            )
            print(
                f"Best validation loss: {best_val_loss:.4f}, Best validation accuracy: {best_val_accuracy:.2f}%"
            )
            break

    # Plot training metrics
    if epoch_numbers:  # Only plot if we have at least one epoch
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Top-left: Training and Validation Loss
        axes[0, 0].plot(
            epoch_numbers, train_losses, label="Training Loss", marker="o", markersize=3
        )
        axes[0, 0].plot(
            epoch_numbers, val_losses, label="Validation Loss", marker="s", markersize=3
        )
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Top-right: Training and Validation Accuracy
        axes[0, 1].plot(
            epoch_numbers,
            train_accuracies,
            label="Training Accuracy",
            marker="o",
            markersize=3,
        )
        axes[0, 1].plot(
            epoch_numbers,
            val_accuracies,
            label="Validation Accuracy",
            marker="s",
            markersize=3,
        )
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy (%)")
        axes[0, 1].set_title("Training and Validation Accuracy")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Bottom-left: Training Loss Only (zoomed)
        axes[1, 0].plot(
            epoch_numbers,
            train_losses,
            label="Training Loss",
            marker="o",
            markersize=3,
            color="tab:blue",
        )
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Loss")
        axes[1, 0].set_title("Training Loss (Zoomed)")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Bottom-right: Validation Loss Only (zoomed)
        axes[1, 1].plot(
            epoch_numbers,
            val_losses,
            label="Validation Loss",
            marker="s",
            markersize=3,
            color="tab:orange",
        )
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Loss")
        axes[1, 1].set_title("Validation Loss (Zoomed)")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("training_metrics.png", dpi=100)
        print("Training metrics plot saved to 'training_metrics.png'")
        plt.close()


def evaluate_model(model, dataset):
    model.eval()
    correct = 0
    total = 0

    # Use DataLoader for batching (handles collation and shapes correctly)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

    with torch.no_grad():
        for feat_pair, IoU_pair in dataloader:
            features_1, features_2 = torch.unbind(feat_pair, dim=1)
            IoU_1, IoU_2 = torch.unbind(IoU_pair, dim=1)
            batch_size = feat_pair.size(0)

            # Get predictions
            pred_1 = model(features_1).squeeze()
            pred_2 = model(features_2).squeeze()

            # Calculate accuracy
            pred_classification = pred_1 > pred_2
            gt_classification = IoU_1 > IoU_2
            total += batch_size
            correct += (pred_classification == gt_classification).sum().item()

    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")


def load_model(model_path="model.pth"):
    """Load a pre-trained model from file"""
    model = ConeClassifier()

    try:
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

        # Handle different save formats
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
            if "accuracy" in checkpoint:
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


def main(mode="train", model_path="model.pth"):
    """
    Main function with mode selection
    mode: 'train' to train a new model, 'eval' to only evaluate existing model
    model_path: path to the saved model file
    """
    # --- Generate dataset from all maps (Map-based split to prevent leakage) ---
    print("Loading and generating dataset contexts...")

    # 1. Split MAP INDICES (not contexts)
    # We have maps 0..N-1 in the loader
    from perceptions.lane_detection.data_loader import generate_data_for_maps, cone_maps

    total_maps = len(cone_maps)
    indices = list(range(total_maps))
    split = int(np.floor(0.7 * total_maps))  # 70% train maps, 30% validation maps

    np.random.seed(42)
    np.random.shuffle(indices)

    train_map_indices = indices[:split]
    val_map_indices = indices[split:]

    print(f"Total Maps: {total_maps}")
    print(f"Training Maps: {train_map_indices}")
    print(f"Validation Maps: {val_map_indices}")

    # 2. Generate Contexts specifically for each split
    # Train: Standard data (Leakage fixed via map split, but no synthetic augmentation)
    print("Generating Training Data (Standard)...")
    train_contexts = generate_data_for_maps(
        train_map_indices,
        perceptual_range=30,
        samples_per_point=1,  # Reverted to 1
        augment_mirror=False,  # Reverted to False
    )

    # Validation: Clean (Single sample, No mirroring - purely evaluation)
    # Validating on mirrored data is arguably good, but let's stick to standard maps first for stability
    print("Generating Validation Data (Clean)...")
    val_contexts = generate_data_for_maps(
        val_map_indices,
        perceptual_range=30,
        samples_per_point=1,  # Clean samples only
        augment_mirror=False,  # No mirroring
    )

    print(f"Training Contexts: {len(train_contexts)}")
    print(f"Validation Contexts: {len(val_contexts)}")

    # 3. Create Datasets
    # Note: Dataset class also has 'augment' flag which adds feature noise.
    # We keep that for training (regularization), disable for val.
    train_dataset = LaneDetectionDataset(
        contexts=train_contexts,
        augment=True,
        false_positive_rate=0.1,
        perceptual_range=30,
        cache_path="perceptions/lane_detection/train_cache.pt",
    )
    val_dataset = LaneDetectionDataset(
        contexts=val_contexts,
        augment=False,
        perceptual_range=30,
        cache_path="perceptions/lane_detection/val_cache.pt",
    )

    if mode == "eval":
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
        train_model(
            train_dataset,
            val_dataset,
            model,
            epochs=250,
            batch_size=1024,
            learning_rate=0.0005,
            L=50,
        )

        # Load the best model and evaluate
        print("Loading best model for final evaluation...")
        best_model = load_model("best_model.pth")
        if best_model:
            print("Final evaluation of best model:")
            evaluate_model(best_model, val_dataset)

        # Save final model
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "final_config": {
                    "input_size": 8,
                    "fc1_size": 256,
                    "fc2_size": 128,
                    "fc3_size": 64,
                    "output_size": 1,
                },
            },
            "model.pth",
        )


if __name__ == "__main__":
    import sys

    # Check command line arguments
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        model_path = sys.argv[2] if len(sys.argv) > 2 else "model.pth"
        print("=== Lane Detection Training Pipeline ===")
        main(mode, model_path)
    else:
        # Default behavior
        print("=== Lane Detection Training Pipeline ===")
        main("train")
