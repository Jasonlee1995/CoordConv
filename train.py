import model

import csv, torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
    
    
def select_model(mode):
    if mode == 'ConvUniform': return model.ConvUniform()
    elif mode == 'ConvQuadrant': return model.ConvQuadrant()
    elif mode == 'CoordConv': return model.CoordConv()
    
    
def approximate_acc(canvas_size, predict_coords, true_coords):
    predict_coords = torch.round(canvas_size * predict_coords).int()
    true_coords = torch.round(canvas_size * true_coords).int()
    return torch.sum(predict_coords == true_coords)
    

class Model():
    def __init__(self, mode, canvas_size, sig_loss=True, gpu_id=0, 
                 save_freq=3, save_csv='./.csv', save_train_fig='./.npy', save_test_fig='./.npy'):
        self.gpu = gpu_id
        self.save_freq = save_freq
        self.save_csv = save_csv
        self.save_train_fig = save_train_fig
        self.save_test_fig = save_test_fig
        
        self.canvas_size = canvas_size
        self.sig_loss = sig_loss

        torch.cuda.set_device(self.gpu)

        self.model = select_model(mode).cuda(self.gpu)
        self.mse = nn.MSELoss().cuda(self.gpu)
        self.sig = nn.BCEWithLogitsLoss().cuda(self.gpu)

    def train(self, train_data, test_data, epochs=10, lr=0.001, weight_decay=0.0005):
        optimizer = optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)
        
        self.model.train()
        for epoch in range(epochs):
            for i, (X, y) in enumerate(train_data):
                X, y = X.float().cuda(self.gpu), y.cuda(self.gpu)
                N, C, H, W = X.size()
                output = self.model(X)
                
                loss = self.mse(output, y)
                if self.sig_loss: loss += self.sig(output, y)

                optimizer.zero_grad(); loss.backward(); optimizer.step()

                if (i+1) % self.save_freq == 0:
                    train_acc = approximate_acc(self.canvas_size, output, y) / (2 * N)
                    test_acc, test_loss = self.test(test_data)
                    train_acc, test_acc = 100*train_acc, 100*test_acc
                    
                    with open(self.save_csv, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([loss.item(), test_loss, train_acc.item(), test_acc.item()])

    def test(self, test_data):
        total, correct = 0, 0
        losses = []

        self.model.eval()
        with torch.no_grad():
            for X, y in test_data:
                X, y = X.float().cuda(self.gpu), y.cuda(self.gpu)
                output = self.model(X)

                total += 2 * y.size(0)
                correct += approximate_acc(self.canvas_size, output, y)
                
                loss = self.mse(output, y)
                if self.sig_loss: loss += self.sig(output, y)
                losses.append(loss.item())
                
        self.model.train()
        return correct/total, sum(losses)/len(losses)
    
    def inference(self, train_data, test_data):
        self.model.eval()
        with torch.no_grad():
            predict_coords = []
            for imgs, _ in train_data:
                imgs = imgs.float().cuda(self.gpu)
                coords = self.model(imgs)
                
                for x, y in coords.cpu().tolist():
                    x, y = round(64 * x), round(64 * y)
                    x, y = max(0, min(self.canvas_size-1, x)), max(0, min(self.canvas_size-1, y))
                    predict_coords.append([x, y])
                    
            predict_statistics = np.array([[0] * self.canvas_size for _ in range(self.canvas_size)])
            for x, y in predict_coords: predict_statistics[y, x] = 255
            np.save(self.save_train_fig, predict_statistics)
        
            predict_coords = []
            for imgs, _ in test_data:
                imgs = imgs.float().cuda(self.gpu)
                coords = self.model(imgs)
                
                for x, y in coords.cpu().tolist():
                    x, y = round(64 * x), round(64 * y)
                    x, y = max(0, min(self.canvas_size-1, x)), max(0, min(self.canvas_size-1, y))
                    predict_coords.append([x, y])
                    
            predict_statistics = np.array([[0] * self.canvas_size for _ in range(self.canvas_size)])
            for x, y in predict_coords: predict_statistics[y, x] = 255
            np.save(self.save_test_fig, predict_statistics)