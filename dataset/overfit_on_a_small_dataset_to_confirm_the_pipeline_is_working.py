small_train_dataset = torch.utils.data.Subset(train_dataset, range(10))
small_train_dataloader = torch.utils.data.DataLoader(small_train_dataset, batch_size=2)

# Train for a few epochs
for epoch in range(5):
    for data in small_train_dataloader:
        inputs = data['input'].to(device)
        labels = data['label'].to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        print(f"Epoch {epoch}, Loss: {loss.item()}")