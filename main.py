import os
import torch
import torch.nn as nn
import torchvision
from dataset import CompoundEyeDataset
from utils.progress_bar import progress_bar
from torchvision.utils import save_image
import random
import wandb

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
f = None
wandb.init(project="test", entity="lovettxh"
    , name="test"
    # ,mode='disabled'
    )

test_loss_all = [[] for _ in range(10)]
test_accuracy_all = [[] for _ in range(10)]

def get_dataloader():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((800, 800)),
        torchvision.transforms.ToTensor(),
    ])
    trainset = CompoundEyeDataset(data_root='../data', train=True, transforms=transform)
    testset = CompoundEyeDataset(data_root='../data', train=False, transforms=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=12, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=12, shuffle=False)
    return train_loader, test_loader

def eval_diff_fog_strength(model, device):
    model.eval()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((800, 800)),
        torchvision.transforms.ToTensor(),
    ])
    testset = CompoundEyeDataset(data_root='../data', train=False, transforms=transform)
    with torch.no_grad():
        for s in range(10):
            testset.set_test_fog_strength(s+1)
            test_loader = torch.utils.data.DataLoader(testset, batch_size=12, shuffle=False)
            err = 0
            loss = 0
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                logit = model(data)
                loss += nn.CrossEntropyLoss()(logit, target).item()
                err += (logit.data.max(1)[1] != target.data).float().sum()
            test_loss_all[s].append(loss/len(test_loader))
            test_accuracy_all[s].append((1-err/len(test_loader.dataset)) * 100)
            # wandb.log({'test loss': loss/len(test_loader), 'test accuracy': (1-err/len(test_loader.dataset)) * 100})
            # print('accuracy: {:.2f}'.format((1-err/len(test_loader.dataset)) * 100), file=f, flush=True)

def eval_dataset(model, data_loader, device):
    model.eval()
    err = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            logit = model(data)
            err += (logit.data.max(1)[1] != target.data).float().sum()
        print('accuracy: {:.2f}'.format((1-err/len(data_loader.dataset)) * 100), file=f, flush=True)

def train(model, train_loader, optimizer):
    model.train()
    loss_sum = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        progress_bar(batch_idx, len(train_loader), 
                    #  'Loss: {:.4f}'.format(loss.item())
                     )
        if (batch_idx+1) % 1000 == 0:
            wandb.log({'train loss': loss_sum/1000})
            loss_sum = 0
            eval_diff_fog_strength(model, device)
            model.train()
    print('Loss: {:.4f}'.format(loss_sum/len(train_loader)), file=f, flush=True)
    
def main():
    train_loader, test_loader = get_dataloader()
    model = torchvision.models.resnet18(num_classes=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
    # model = torchvision.models.wide_resnet101_2()

    for epoch in range(2):
        print(f'Epoch: {epoch}')
        train(model, train_loader, optimizer)
        schedular.step()
        eval_dataset(model, train_loader, device)
        eval_dataset(model, test_loader, device)
    wandb.log({"test loss": wandb.plot.line_series(xs=range(len(test_loss_all[0])),
                                                   ys=test_loss_all, 
                                                   keys=["strength{}".format(i) for i in range(1, 11)],
                                                   title="test loss",
                                                   xname="fog strength")})
    
    wandb.log({"test accuracy": wandb.plot.line_series(xs=range(len(test_accuracy_all[0])),
                                                   ys=test_accuracy_all, 
                                                   keys=["strength{}".format(i) for i in range(1, 11)],
                                                   title="test accuracy",
                                                   xname="fog strength")})
    # torch.save(model.state_dict(), './model-checkpoint/resnet18.pt')

    # for data, target in test_loader:
    #     save_image(data[5], 'test.png')
    #     print(target[5])
    #     break

if __name__ == '__main__':
    main()