import itertools
import os
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, TensorDataset

def load_cifar100_subset(target_classes, train_transform, val_transform, root='./cifar_100'):
    cifar100_path = os.path.join(root, 'cifar-100-python')
    if os.path.isdir(cifar100_path):
        print(f"Dataset found in '{root}'. Loading from local files.")
    else:
        print(f"Dataset not found in '{root}'. Downloading...")
    train_dataset_full = torchvision.datasets.CIFAR100(
        root=root, 
        train=True, 
        download=True, 
        transform=train_transform
    )
    test_dataset_full = torchvision.datasets.CIFAR100(
        root=root, 
        train=False, 
        download=True, 
        transform=val_transform
    )
    print("Dataset loaded successfully.")
    all_classes = train_dataset_full.classes
    try:
        target_indices = [all_classes.index(cls) for cls in target_classes]
    except ValueError as e:
        print(f"Error: One of the target classes not found in CIFAR-100. {e}")
        return None, None
    label_map = {old_label: new_label for new_label, old_label in enumerate(target_indices)}
    def _filter_dataset(dataset):
        targets_np = np.array(dataset.targets)
        indices_to_keep = np.isin(targets_np, target_indices)
        dataset.data = dataset.data[indices_to_keep]
        original_targets_to_keep = targets_np[indices_to_keep]
        dataset.targets = [label_map[target] for target in original_targets_to_keep]
        dataset.classes = target_classes
        return dataset
    print(f"Filtering for {len(target_classes)} classes...")
    train_dataset_subset = _filter_dataset(train_dataset_full)
    test_dataset_subset = _filter_dataset(test_dataset_full)
    print("Filtering complete. Returning training and validation datasets.")
    return train_dataset_subset, test_dataset_subset

def visualise_images(loader, grid):
    rows, cols = grid
    num_images_to_show = rows * cols
    dataset_to_show = loader.dataset
    class_indices = defaultdict(list)
    for idx, target in enumerate(dataset_to_show.targets):
        class_indices[target].append(idx)
    class_names = dataset_to_show.classes
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    for i, ax in enumerate(axes.flat):
        if i >= num_images_to_show or i >= len(class_names):
            ax.axis('off')
            continue
        class_label = i
        indices_for_class = class_indices[class_label]
        if not indices_for_class:
            ax.axis('off')
            continue
        random_image_index = random.choice(indices_for_class)
        image_tensor, _ = dataset_to_show[random_image_index]
        img_to_display = image_tensor.numpy().transpose((1, 2, 0))
        min_val = img_to_display.min()
        max_val = img_to_display.max()
        img_to_display = (img_to_display - min_val) / (max_val - min_val)
        class_name = class_names[class_label]
        ax.imshow(img_to_display)
        ax.set_title(class_name.capitalize(), fontsize=16)
        ax.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

def plot_training_metrics(metrics):
    train_losses, val_losses, val_accuracies = metrics
    num_epochs = len(train_losses)
    epochs = range(1, num_epochs + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax1 = axes[0]
    ax1.plot(epochs, train_losses, color='#085c75', linewidth=2.5, marker='o', markersize=5, label='Training Loss')
    ax1.plot(epochs, val_losses, color='#fa5f64', linewidth=2.5, marker='o', markersize=5, label='Validation Loss')
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax2 = axes[1]
    ax2.plot(epochs, val_accuracies, color='#fa5f64', linewidth=2.5, marker='o', markersize=5, label='Validation Accuracy')
    ax2.set_title('Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    x_interval = (num_epochs - 1) // 10 + 1
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=1, right=num_epochs)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(x_interval))
        ax.tick_params(axis='both', which='major', labelsize=10)
    plt.tight_layout()
    plt.show()

def verify_training_process(model_class, train_loader, loss_function, train_epoch_fn, device):
    print("Verifying train_epoch")
    NUM_VERIFY_EPOCHS = 5
    NUM_VERIFY_BATCHES = 10
    verify_model = model_class(15).to(device)
    verify_optimizer = optim.Adam(verify_model.parameters(), lr=0.0005)
    batches = list(itertools.islice(iter(train_loader), NUM_VERIFY_BATCHES))
    all_images = torch.cat([b[0] for b in batches])
    all_labels = torch.cat([b[1] for b in batches])
    verify_subset_dataset = TensorDataset(all_images, all_labels)
    verify_subset_loader = DataLoader(verify_subset_dataset, batch_size=train_loader.batch_size)
    initial_weight = verify_model.conv_block1.block[0].weight.clone()
    epoch_losses = []
    print(f"Training on {len(verify_subset_dataset)} images for {NUM_VERIFY_EPOCHS} epochs:")
    for epoch in range(NUM_VERIFY_EPOCHS):
        loss = train_epoch_fn(
            model=verify_model,
            train_loader=verify_subset_loader,
            loss_function=loss_function,
            optimizer=verify_optimizer,
            device=device
        )
        epoch_losses.append(loss)
        print(f"Epoch [{epoch+1}/{NUM_VERIFY_EPOCHS}], Loss: {loss:.4f}")
    trained_weight = verify_model.conv_block1.block[0].weight
    weights_changed = not torch.equal(initial_weight, trained_weight)
    if weights_changed:
        print("Weight Update Check: Model weights changed during training.")
    else:
        print("Weight Update Check: Model weights DID NOT change.")
    loss_decreased = epoch_losses[-1] < epoch_losses[0]
    if loss_decreased:
        print(f"Loss Trend Check: Loss decreased from {epoch_losses[0]:.4f} to {epoch_losses[-1]:.4f}.")
    else:
        print("Loss Trend Check: Loss DID NOT show a decreasing trend.")

def verify_validation_process(model_class, val_loader, loss_function, validate_epoch_fn, device):
    print("Verifying validate_epoch")
    NUM_VERIFY_BATCHES = 10
    verify_model = model_class(15).to(device)
    val_batches = list(itertools.islice(iter(val_loader), NUM_VERIFY_BATCHES))
    val_all_images = torch.cat([b[0] for b in val_batches])
    val_all_labels = torch.cat([b[1] for b in val_batches])
    verify_val_subset_dataset = TensorDataset(val_all_images, val_all_labels)
    verify_val_subset_loader = DataLoader(verify_val_subset_dataset, batch_size=val_loader.batch_size)
    initial_weight = verify_model.conv_block1.block[0].weight.clone()
    print(f"Validating on {len(verify_val_subset_dataset)} images:")
    val_loss, val_accuracy = validate_epoch_fn(
        model=verify_model,
        val_loader=verify_val_subset_loader,
        loss_function=loss_function,
        device=device
    )
    validated_weight = verify_model.conv_block1.block[0].weight
    print(f"Returned Validation Loss: {val_loss:.4f}")
    print(f"Returned Validation Accuracy: {val_accuracy:.2f}%")
    types_correct = isinstance(val_loss, float) and isinstance(val_accuracy, float)
    if types_correct:
        print("Return Types Check: Function returned a float for loss and accuracy.")
    else:
        print("Return Types Check: Function DID NOT return the correct data types.")
    weights_unchanged = torch.equal(initial_weight, validated_weight)
    if weights_unchanged:
        print("Weight Integrity Check: Model weights were not changed during validation.")
    else:
        print("Weight Integrity Check: Model weights WERE CHANGED.")