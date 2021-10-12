import dataset, train

import argparse, torch
import torchvision.transforms as transforms

import warnings
warnings.filterwarnings('ignore')


# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--count', type=int, default=1)

parser.add_argument('--train_csv', type=str)
parser.add_argument('--test_csv', type=str)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--canvas_size', type=int, default=64)

parser.add_argument('--mode', type=str)
parser.add_argument('--sig_loss', type=bool)
parser.add_argument('--gpu_id', type=int)
parser.add_argument('--save_csv', type=str)
parser.add_argument('--save_train_fig', type=str)
parser.add_argument('--save_test_fig', type=str)

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float)
parser.add_argument('--weight_decay', type=float)

args = parser.parse_args()


# dataset settings
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = dataset.RegressionDataset(args.train_csv, args.canvas_size, transform)
test_dataset = dataset.RegressionDataset(args.test_csv, args.canvas_size, transform)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)


# train models and save
for i in range(args.count):
    save_csv = args.save_csv + str(i).zfill(4) + '.csv'
    save_train_fig = args.save_train_fig + str(i).zfill(4) + '.npy'
    save_test_fig = args.save_test_fig + str(i).zfill(4) + '.npy'
    
    model = train.Model(args.mode, args.canvas_size, sig_loss=args.sig_loss, gpu_id=args.gpu_id, 
                        save_csv=save_csv, save_train_fig=save_train_fig, save_test_fig=save_test_fig)
    
    model.train(trainloader, testloader, args.epochs, args.lr, args.weight_decay)
    model.inference(trainloader, testloader)