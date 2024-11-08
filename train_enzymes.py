import torch
import torch_geometric as pyg
from gcn import GCNGraphLev
from gin import GINGraphLev
from gat import GATGraphLev
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device {device}')

def train(model, dataloader, criterion, optimizer):
    model.train()
    for data in dataloader:  # Iterate in batches over the training dataset.
         data = data.to(device)
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def get_acc(model, dataloader, criterion):
     model.eval()
     correct = 0
     for data in dataloader:  # Iterate in batches over the training/test dataset.
         data = data.to(device)
         out = model(data.x, data.edge_index, data.batch)  
         loss = criterion(out, data.y) 
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return loss, (correct / len(dataloader.dataset))  # Derive ratio of correct predictions.

def main():
    dataset = pyg.datasets.TUDataset(root='data/TUDataset', name='ENZYMES', force_reload=True)

    dataset = dataset.shuffle()
    train_dataset = dataset[:420]
    val_dataset = dataset[420:510]
    test_dataset = dataset[510:]
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    d_in, d_lat, d_out = dataset.num_node_features, 64, dataset.num_classes
    gcn = GCNGraphLev(d_in, d_lat, d_out).to(device)
    gin = GINGraphLev(d_in, d_lat, d_out).to(device)
    gat = GATGraphLev(d_in, d_lat, d_out).to(device)
    train_acc_list = []
    test_acc_list = []
    for model in [gcn, gin, gat]: # add gat
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss()

        val_loss_list = []
        val_model_params = []
        model.train()
        patience_counter = 0

        for epoch in tqdm(range(300)):
            train(model, train_loader, criterion, optimizer)
            val_loss, _ = get_acc(model, val_loader, criterion)
            tqdm.print(f'Epoch {epoch}. Val Loss = {val_loss}')

            if((len(val_loss_list) >= 30) and (val_loss > min(val_loss_list))):
                patience_counter += 1
                if(patience_counter > 15):
                    best_model_state = val_model_params[val_loss_list.index(min(val_loss_list))]
                    model.load_state_dict(best_model_state)
                    print(f'Early Stopping at Epoch {epoch}')
                    break
            else:
                patience_counter = 0
                val_loss_list.append(val_loss)
                val_model_params.append(model.state_dict())

        train_acc_list.append(get_acc(model, train_loader, criterion)[1])
        test_acc_list.append(get_acc(model, test_loader, criterion)[1])

    print(train_acc_list, test_acc_list)


if(__name__ == '__main__'):
    main()