import os
import torch
import torch.nn as nn
import torchvision
from dataset import CompoundEyeDataset
from utils.progress_bar import progress_bar
from torchvision.utils import save_image

os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

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

def eval_dataset(model, data_loader, device, f=None):
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
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()
        optimizer.step()
        progress_bar(batch_idx, len(train_loader))

    
def main():
    train_loader, test_loader = get_dataloader()
    model = torchvision.models.resnet18(num_classes=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)
    # model = torchvision.models.wide_resnet101_2()

    for epoch in range(10):
        print(f'Epoch: {epoch}')
        train(model, train_loader, optimizer)
        schedular.step()
        eval_dataset(model, train_loader, device)
        eval_dataset(model, test_loader, device)

    torch.save(model.state_dict(), './model-checkpoint/resnet50.pt')

    # for data, target in test_loader:
    #     save_image(data[5], 'test.png')
    #     print(target[5])
    #     break

if __name__ == '__main__':
    main()