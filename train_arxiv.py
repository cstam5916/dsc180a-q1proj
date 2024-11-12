import torch
import torch_geometric as pyg
from gcn import GCN
from gin import GIN
from gat import GAT
from smpnn import SMPNN
from tqdm.auto import tqdm
import torch.nn.functional as F
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import DataLoader
from torch_geometric.loader import RandomNodeLoader
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    torch.autograd.set_detect_anomaly(True)
    dataset = PygNodePropPredDataset(name="ogbn-arxiv", root='dataset/')
    split_idx = dataset.get_idx_split()

    data = dataset[0]
    for split in ['train', 'valid', 'test']:
        mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        mask[split_idx[split]] = True
        data[f'{split}_mask'] = mask

    train_loader = RandomNodeLoader(data, num_parts=40, shuffle=True, num_workers=5)
    test_loader = RandomNodeLoader(data, num_parts=5, num_workers=5)

    d_in, d_lat, d_out = dataset.num_node_features, 64, dataset.num_classes
    gcn = GCN(d_in, d_lat, d_out).to(device)
    gin = GIN(d_in, d_lat, d_out).to(device)
    gat = GAT(d_in, d_lat, d_out, heads=8).to(device)
    smpnn = SMPNN(d_in, d_lat, d_out, num_layers=6).to(device)

    outer_loss_list = []

    for model in [gcn, gin, gat, smpnn]:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        val_loss_list = []
        val_model_params = []
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 5  # Stop after 5 epochs without improvement

        model.train()
        for epoch in tqdm(range(300)):
            model.train()
            train_loss_list = []
            valid_loss_list = []
            
            for batch in train_loader:
                model.train()
                optimizer.zero_grad()
                batch = batch.to(device)
                out = model(batch)
                loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask].squeeze(-1))
                loss.backward()
                optimizer.step()
                train_loss_list.append(loss)  # Append the tensor directly

            model.eval()
            valid_loss_list = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    out = model(batch)
                    val_loss = F.nll_loss(out[batch.valid_mask], batch.y[batch.valid_mask].squeeze(-1))
                    valid_loss_list.append(val_loss)
            mean_val_loss = torch.stack(valid_loss_list).mean().item()
            mean_train_loss = torch.stack(train_loss_list).mean().item()
            val_loss_list.append(mean_val_loss)

            tqdm.write(f'Epoch {epoch} training loss: {mean_train_loss}, val loss: {mean_val_loss}')

            if mean_val_loss < best_val_loss:
                best_val_loss = mean_val_loss
                patience_counter = 0
                best_model_params = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    tqdm.write(f'Early stopping at epoch {epoch} with best val loss: {best_val_loss}')
                    model.load_state_dict(best_model_params)
                    outer_loss_list.append(val_loss_list)
                    break
    for name, llist in zip(['GCN', 'GIN', 'GAT', 'SMPNN'], outer_loss_list):
        plt.plot(range(len(llist)), llist, label=name)
    plt.legend()
    plt.xlabel('Epoch (early stopping enabled)')
    plt.ylabel('Validation Loss')
    plt.title('Training Curves for 4 GNN models on OGB Arxiv Data')
    plt.savefig('figs/ogbn-arxiv-val-loss.pdf')


if __name__ == '__main__':
    print(f'Using device {device}')
    main()