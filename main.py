import os
import argparse
import torch
import torch.nn as nn
import torchvision
import pandas
from dataset import CompoundEyeDataset
from utils.progress_bar import progress_bar

parser = argparse.ArgumentParser(description='Compound Eye Classification Experiment') 
parser.add_argument('--data_dir', default='../data',type=str)
args = parser.parse_args()

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


training_loss_all = []
test_loss_all = [[] for _ in range(10)]
test_accuracy_all = [[] for _ in range(10)]

# get dataloader for training and testing
# training and testing data are randomly split with a ratio of 9:1
def get_dataloader():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((800, 800)),
        torchvision.transforms.ToTensor(),
    ])
    trainset = CompoundEyeDataset(data_root=args.data_dir, train=True, transforms=transform)
    testset = CompoundEyeDataset(data_root=args.data_dir, train=False, transforms=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=12, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=12, shuffle=False)
    return train_loader, test_loader

# evaluate model accuracy on different fog strength 
# images have been manually divided into 10 different fog strength
def eval_diff_fog_strength(model, device):
    model.eval()
    transform = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((800, 800)),
        torchvision.transforms.ToTensor(),
    ])
    testset = CompoundEyeDataset(data_root=args.data_dir, train=False, transforms=transform)
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
            # print('accuracy: {:.2f}'.format((1-err/len(test_loader.dataset)) * 100), file=f, flush=True)

# model evaluation on training/testing dataset
def eval_dataset(model, data_loader, device):
    model.eval()
    err = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            logit = model(data)
            err += (logit.data.max(1)[1] != target.data).float().sum()
        print('accuracy: {:.2f}'.format((1-err/len(data_loader.dataset)) * 100), file=f, flush=True)

# model training process
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
        
        # fog strength evaluation is done every 1000 batches
        if (batch_idx+1) % 1000 == 0:
            training_loss_all.append(loss_sum/1000)
            loss_sum = 0
            eval_diff_fog_strength(model, device)
            model.train()
    print('Loss: {:.4f}'.format(loss_sum/len(train_loader)), file=f, flush=True)
    
def main():
    train_loader, test_loader = get_dataloader()
    # ResNet18 is selected as model backbone
    model = torchvision.models.resnet18(num_classes=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    schedular = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 10)

    # training process
    for epoch in range(2):
        print(f'Epoch: {epoch}')
        train(model, train_loader, optimizer)
        schedular.step()
        eval_dataset(model, train_loader, device)
        eval_dataset(model, test_loader, device)

    # save model
    if not os.path.exists('./model-checkpoint'):
        os.makedirs('./model-checkpoint')
    torch.save(model.state_dict(), './model-checkpoint/resnet18_.pt')
    # save training and testing results
    if not os.path.exists('./results'):
        os.makedirs('./results')
    
    df_train_loss = pandas.DataFrame(training_loss_all)
    df_test_loss = pandas.DataFrame(test_loss_all).T
    df_test_accuracy = pandas.DataFrame(test_accuracy_all).T
    
    df_train_loss.columns = ['training_loss']
    df_test_accuracy.columns = ['strength_'+str(i+1) for i in range(10)]
    df_test_loss.columns = ['strength_'+str(i+1) for i in range(10)]

    df_train_loss.insert(0, 'Step', range(1, len(training_loss_all)+1))
    df_test_loss.insert(0, 'Step', range(1, len(training_loss_all)+1))
    df_test_accuracy.insert(0, 'Step', range(1, len(training_loss_all)+1))

    df_train_loss.to_csv('./results/training_loss.csv', index=False)
    df_test_loss.to_csv('./results/test_loss.csv', index=False)
    df_test_accuracy.to_csv('./results/test_accuracy.csv', index=False)

if __name__ == '__main__':
    main()