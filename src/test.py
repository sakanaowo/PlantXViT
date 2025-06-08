device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PlantXViT(num_classes=4).to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

history = train_model(
    model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=50,
    device=device,
    save_path=config["output"]["model_path"]
)
