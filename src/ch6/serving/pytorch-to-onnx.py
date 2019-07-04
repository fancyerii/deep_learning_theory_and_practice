import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearRegression(1, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

from torch.utils import data


class FakeDate(data.Dataset):
    def __init__(self, total):
        self.x_train = np.random.uniform(-10, 10, total). \
            reshape([-1, 1]).astype(np.float32)
        self.y_train = self.x_train * 2.0 + 5 + \
                       np.random.normal(0, 0.1, size=[total, 1]).astype(np.float32)

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]


# Training the Model
num_epochs = 100
batch_size = 10
training_set = FakeDate(100)
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': 1}
training_generator = data.DataLoader(training_set, **params)
for epoch in range(num_epochs):
    for i, data in enumerate(training_generator):
        x_batch, y_batch = data
        optimizer.zero_grad()
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(training_set) // batch_size, loss.data[0]))
w = list(model.parameters())
print(w)
import torch.onnx

input = Variable(torch.zeros(1, 1, dtype=torch.float))
model_path = "lr.onnx"
torch.onnx.export(model, input, model_path)

