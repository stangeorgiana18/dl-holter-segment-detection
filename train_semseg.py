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

def main(yaml_file, test_mode=False):
    ######### Prepare Environment ###########
    # Select device: GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load configuration settings from the YAML file
    opt = load_yaml(yaml_file, saveYaml2output=True)
    epoch = opt.OPTIM.NUM_EPOCHS

    # Directory for saving models
    model_dir = opt.SAVE_DIR + 'models/'
    dir_utils.mkdir_with_del(model_dir)

    ######### Dataset ###########
    # Initialize training dataset
    try:
        train_dataset = dataloader.Noise_Dataset(
            opt.DATASET.TRAIN_CSV,
            leads=opt.DATASET_CUSTOME.LEADS,
            date_len=opt.DATASET_CUSTOME.INPUT_LENGTH,
            n_max_cls=opt.DATASET_CUSTOME.OUT_C,
            random_crop=True,
            transform=dataloader.get_transform(train=True)
        )
    except KeyError as e:
        print(f"Error loading train dataset: Missing column {e} in CSV file.")
        return

    # Initialize validation dataset
    try:
        val_dataset = dataloader.Noise_Dataset(
            opt.DATASET.VAL_CSV,
            leads=opt.DATASET_CUSTOME.LEADS,
            date_len=opt.DATASET_CUSTOME.INPUT_LENGTH,
            n_max_cls=opt.DATASET_CUSTOME.OUT_C,
            random_crop=False,
            transform=dataloader.get_transform(train=False)
        )
    except KeyError as e:
        print(f"Error loading validation dataset: Missing column {e} in CSV file.")
        return

    # Create data loaders for training and validation datasets
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=False, num_workers=4
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
        num_heads=opt.MODEL.N_HEADS
    ).to(device)

    ######### Optimizer and Scheduler ###########
    # Define optimizer with initial learning rate and weight decay
    optimizer = optim.Adam(
        model.parameters(), lr=opt.OPTIM.LR_INITIAL, betas=(0.9, 0.999), eps=1e-8
    )

    # Learning rate scheduler with warm-up and cosine annealing
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, opt.OPTIM.NUM_EPOCHS - warmup_epochs, eta_min=opt.OPTIM.LR_MIN
    )
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

    # Define loss function with class weights
    criterion = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([1, 1, 1]).to(device))

    # Enable mixed precision training for faster computation
    grad_scaler = torch.amp.GradScaler()

    ######### Training Loop ###########
    for epoch in range(1, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        epoch_train_loss = 0
        for i, data in enumerate(tqdm(train_dataloader), 0):
            inputs = data['input'].to(device)
            labels = data['label'].to(device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass with automatic mixed precision
            with torch.amp.autocast(device_type='cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            # Backward pass and optimizer step
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            # Accumulate training loss
            epoch_train_loss += loss.item() * inputs.size(0)

        # Calculate mean training loss
        train_loss_mean = epoch_train_loss / dataset_sizes['train']

        # Validation phase
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for data in val_dataloader:
                inputs = data['input'].to(device)
                labels = data['label'].to(device)

                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item() * inputs.size(0)

        # Calculate mean validation loss
        val_loss_mean = epoch_val_loss / dataset_sizes['val']

        # Update the learning rate scheduler
        scheduler.step()

        # Save the model checkpoint
        save_path = f"{model_dir}model_epoch_{epoch}_val_{val_loss_mean:.6f}.pth"
        torch.save(model, save_path)
        print(f"Model saved to {save_path}")

        # Log training and validation results
        print("------------------------------------------------------------------")
        print(f"Epoch: {epoch} | Time: {time.time() - epoch_start_time:.4f}s | Train Loss: {train_loss_mean:.6f} | Val Loss: {val_loss_mean:.6f}")
        print("------------------------------------------------------------------")

if __name__ == "__main__":
    from argparse import ArgumentParser

    # Parse command-line arguments for configuration file
    parser = ArgumentParser(description="Train Semantic Segmentation Model")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to YAML config file")
    args = parser.parse_args()

    # Run the main training function
    main(args.config, test_mode=False)
