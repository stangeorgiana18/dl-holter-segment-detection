import torch
import numpy as np
import torch.optim as optim
from torch.cuda import amp
from tqdm import tqdm
import time
from dataset import dataloader_semseg as dataloader
from tools import dir_utils
from configs.load_yaml import load_yaml
from tools.lr_warmup_scheduler import GradualWarmupScheduler

def calculate_accuracy(predictions, labels):
    """
    Calculate accuracy by comparing predictions and ground truth labels.

    Args:
        predictions (numpy array): Predicted class indices.
        labels (numpy array): Ground truth class indices.

    Returns:
        float: Accuracy as a fraction of correct predictions.
    """
    # Flatten predictions and labels for comparison
    predictions = predictions.flatten()
    labels = labels.flatten()

    # Calculate the number of correct predictions
    correct = (predictions == labels).sum()
    total = labels.size  # Total number of elements
    accuracy = correct / total
    print(f"Correct: {correct}, Total: {total}, Accuracy: {accuracy}")
    return accuracy

def main(yaml_file, test_mode=False):
    """
    Main training function.

    Args:
        yaml_file (str): Path to the YAML configuration file.
        test_mode (bool): Whether to run in test mode (default: False).
    """
    # Set the device to GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load configuration settings from the YAML file
    opt = load_yaml(yaml_file, saveYaml2output=True)
    epoch = opt.OPTIM.NUM_EPOCHS

    # Directory to save trained models
    model_dir = opt.SAVE_DIR + 'models/'
    dir_utils.mkdir_with_del(model_dir)

    ######### Dataset ###########
    # Create training and validation datasets
    train_dataset = dataloader.Noise_Dataset(
        opt.DATASET.TRAIN_CSV,
        leads=opt.DATASET_CUSTOME.LEADS,
        date_len=opt.DATASET_CUSTOME.INPUT_LENGTH,
        n_max_cls=opt.DATASET_CUSTOME.OUT_C,
        random_crop=True,
        transform=dataloader.get_transform(train=True),
    )

    val_dataset = dataloader.Noise_Dataset(
        opt.DATASET.VAL_CSV,
        leads=opt.DATASET_CUSTOME.LEADS,
        date_len=opt.DATASET_CUSTOME.INPUT_LENGTH,
        n_max_cls=opt.DATASET_CUSTOME.OUT_C,
        random_crop=False,
        transform=dataloader.get_transform(train=False),
    )

    # Create data loaders for batch processing
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.OPTIM.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        prefetch_factor=3,
        persistent_workers=False,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=opt.OPTIM.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        prefetch_factor=3,
        persistent_workers=False,
    )

    dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
    print('===> Loading datasets done')

    ######### Model ###########
    # Import and initialize the model
    from models.OneD_CNN_Unet import Model

    model = Model(
        in_c=1,
        out_c=opt.DATASET_CUSTOME.OUT_C,
        img_size=opt.DATASET_CUSTOME.INPUT_LENGTH,
        embed_dim=opt.MODEL.EMBED_DIM,
        patch_size=opt.MODEL.PATCH_SIZE,
        window_size=opt.MODEL.WINDOW_SIZE,
        depths=opt.MODEL.DEPTHS,
        num_heads=opt.MODEL.N_HEADS,
    ).to(device)

    ######### Optimizer and Scheduler ###########
    # Set up the optimizer and learning rate scheduler
    new_lr = opt.OPTIM.LR_INITIAL
    optimizer = optim.Adam(
        model.parameters(),
        lr=new_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-5,
    )

    # Define a warm-up scheduler followed by a cosine annealing scheduler
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs, eta_min=opt.OPTIM.LR_MIN
    )
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine
    )
    scheduler.step()

    ######### Loss Function ###########
    # Define the loss function with class weights
    n_classes = opt.DATASET_CUSTOME.OUT_C
    weights = torch.ones(n_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)

    # Enable automatic mixed precision for faster training
    grad_scaler = amp.GradScaler()
    start_epoch = 0

    ######### Training and Validation ###########
    for epoch in range(1, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        epoch_train_loss = 0
        train_outputs_all, train_labels_all = [], []

        for data in tqdm(train_dataloader):
            # Transfer input and labels to the device (GPU or CPU)
            inputs = data['input'].to(device)
            labels = data['label'].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass with automatic mixed precision
            with amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward pass and optimizer step
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            # Accumulate training loss and predictions
            epoch_train_loss += loss.item() * inputs.size(0)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            train_outputs_all.extend(predictions)
            train_labels_all.extend(labels.cpu().numpy())

        # Calculate mean training loss and accuracy
        train_loss_mean = epoch_train_loss / dataset_sizes['train']
        train_accuracy = calculate_accuracy(np.array(train_outputs_all), np.array(train_labels_all))

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        val_outputs_all, val_labels_all = [], []

        with torch.no_grad():
            for data in val_dataloader:
                inputs = data['input'].to(device)
                labels = data['label'].to(device)

                # Forward pass and calculate loss
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item() * inputs.size(0)

                predictions = outputs.argmax(dim=1).cpu().numpy()
                val_outputs_all.extend(predictions)
                val_labels_all.extend(labels.cpu().numpy())

        # Calculate mean validation loss and accuracy
        val_loss_mean = epoch_val_loss / dataset_sizes['val']
        val_accuracy = calculate_accuracy(np.array(val_outputs_all), np.array(val_labels_all))

        # Update the learning rate scheduler
        scheduler.step()

        # Save the model checkpoint
        save_path = model_dir + 'model_epoch_{}_val_{:.6f}.pth'.format(epoch, val_loss_mean)
        torch.save(model.state_dict(), save_path)

        # Log training and validation results
        print("------------------------------------------------------------------")
        print(f"Epoch: {epoch}\tTime: {time.time() - epoch_start_time:.4f}s\t"
              f"Train Loss: {train_loss_mean:.6f}\tVal Loss: {val_loss_mean:.6f}\t"
              f"Train Acc: {train_accuracy:.4f}\tVal Acc: {val_accuracy:.4f}\t"
              f"Learning Rate: {scheduler.get_lr()[0]:.8f}")
        print("------------------------------------------------------------------")

if __name__ == "__main__":
    from argparse import ArgumentParser

    # Parse command-line arguments
    parser = ArgumentParser(description="train")
    parser.add_argument("-c", "--config", type=str, default=None, help="path to yaml file")
    args = parser.parse_args()

    # Run the main training function
    main(args.config, test_mode=False)
