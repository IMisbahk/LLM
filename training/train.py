import torch
import torch.optim as optim
from torch.utils.data import DataLoader

def train(model, dataset, epochs=10, batch_size=32):
    model.train()  # Set the model to training mode
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for inputs in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)

            # Reshape outputs and targets for loss computation
            loss = criterion(outputs.view(-1, outputs.size(-1)), inputs.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {avg_loss}')
