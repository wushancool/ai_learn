import torch
from tqdm import tqdm

def evaluate_accuracy(model, dataloader, device):
    training = model.training
    model.eval()

    corrects, total = 0,0
    for x, y in dataloader:
        with torch.no_grad():
            x,y = x.to(device), y.to(device)
            y_hat = model(x)
            corrects += (y_hat.argmax(-1) == y).sum().item()
            total += len(y)
    
    model.train(training)
    return corrects/total

def train_step(model, x, y, optimizer, criterion):
    model.train()
    y_hat = model(x)

    optimizer.zero_grad()
    loss = criterion(y_hat, y)
    loss.sum().backward()
    optimizer.step()

    corrects = (y_hat.argmax(-1) == y).sum().item()

    return loss.sum().item(), corrects


def train(model, optimizer, criterion, train_data, test_data, epochs, device):
    model.to(device)
    
    total_loss, total_corrects, total_count = 0,0,0
    for epoch in range(epochs):
        with tqdm(total=len(train_data)) as p:
            for x, y in train_data:
                x,y = x.to(device), y.to(device)
                loss, corrects = train_step(model, x, y, optimizer, criterion)
                total_loss+=loss
                total_corrects+=corrects
                total_count+=len(y)

                p.update()
                p.set_description(f"epoch {epoch+1}/{epochs}, loss:{total_loss/total_count:.2f}, "
                                    f"acc:{100*total_corrects/total_count:.2f}%")
                                
        val_acc = evaluate_accuracy(model, test_data, device)
        print(f"Validate accuracy at epoch {epoch+1}: {100*val_acc:.2f}%")