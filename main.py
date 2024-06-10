import torch
import torchvision
from dataset import CompoundEyeDataset


def main():
    trainset = CompoundEyeDataset(data_root='../data', train=True)
    testset = CompoundEyeDataset(data_root='../data', train=False)

    print(trainset.test_list == testset.test_list)
if __name__ == '__main__':
    main()