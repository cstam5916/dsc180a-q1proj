import torch
import torch_geometric as pyg
from gcn import GCN
from gin import GIN
from gat import GAT
from smpnn import SMPNN
from tqdm.auto import tqdm
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device {device}')

def main():
    torch.autograd.set_detect_anomaly(True)
    dataset = pyg.datasets.Planetoid(root='/tmp/Cora', name='Cora').to(device)
    d_in, d_lat, d_out = dataset.num_node_features, 64, dataset.num_classes
    gcn = GCN(d_in, d_lat, d_out).to(device)
    gin = GIN(d_in, d_lat, d_out).to(device)
    gat = GAT(d_in, d_lat, d_out, heads=16).to(device)
    smpnn = SMPNN(d_in, d_lat, d_out, num_layers=12).to(device)
    train_acc_list = []
    test_acc_list = []
    for model in [gcn, smpnn]: # add gat
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        val_loss_list = []
        val_model_params = []
        model.train()
        for epoch in tqdm(range(300)):
            model.train()
            optimizer.zero_grad()
            out = model(dataset)
            loss = F.nll_loss(out[dataset.train_mask], dataset.y[dataset.train_mask])
            loss.backward()
            optimizer.step()

            if(epoch % 10 == 0):
                model.eval()
                out = model(dataset)
                val_loss = F.nll_loss(out[dataset.val_mask], dataset.y[dataset.val_mask]).item()
                if((len(val_loss_list) >= 5) and all(x < val_loss for x in val_loss_list[-5:])):
                    best_model_state = val_model_params[val_loss_list.index(min(val_loss_list))]
                    model.load_state_dict(best_model_state)
                    print(f'Early Stopping at Epoch {epoch}')
                    break
                else:
                    val_loss_list.append(val_loss)
                    val_model_params.append(model.state_dict())
        model.eval()
        pred = model(dataset).argmax(dim=1)
        train_correct = (pred[dataset.train_mask] == dataset.y[dataset.train_mask]).sum()
        train_acc = int(train_correct) / int(dataset.train_mask.sum())
        train_acc_list.append(train_acc)

        test_correct = (pred[dataset.test_mask] == dataset.y[dataset.test_mask]).sum()
        test_acc = int(test_correct) / int(dataset.test_mask.sum())
        test_acc_list.append(test_acc)

    print(train_acc_list, test_acc_list)


if(__name__ == '__main__'):
    main()