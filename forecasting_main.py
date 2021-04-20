import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#cd /home/lut/Desktop/slicing_powersystems


import numpy
data1 = numpy.loadtxt('2019-04-01_00h_UTC_PMUID01.txt')
data2 = numpy.loadtxt('2019-04-01_00h_UTC_PMUID02.txt')


size_dataset = 500;
mag_VA1 = data1[:size_dataset,6];
mag_IA1 = data1[:size_dataset,8];
mag_VA2 = data2[:size_dataset,6];
mag_IA2 = data2[:size_dataset,8];
import matplotlib.pyplot as plt
plt.plot(mag_VA1)
plt.plot(mag_VA2)


plt.plot(mag_IA1)
plt.plot(mag_IA2)

info1 = numpy.column_stack([mag_VA1,mag_IA1])
info2 = numpy.column_stack([mag_VA2,mag_IA2])

unos = numpy.ones(info1.shape[0]);
unos = unos.reshape(info1.shape[0],1);
label_1 = unos * 1;
label_2 = unos * 2;


info1 = numpy.column_stack([info1, label_1])
info2 = numpy.column_stack([info2, label_2])

size_iteration = info1.shape[0];
dataset = info1[:size_iteration, ] * 0;
index1 = 0;
index2 = 0;
for ii in range(size_iteration):

    # TO GET A RANDOM GENERATION OF EVENT 2
    s = numpy.random.uniform(0, 1, 1)
    if s > 0.5:
        step = 3
    elif s < 0.75:
        step = 5
    else:
        step = 7

    # Generation of elastic dataset
    trigger = numpy.remainder(ii, 10);
    if trigger == step:
        dataset[ii,] = info2[index2,]
        index2 = index2 + 1;
    else:
        dataset[ii,] = info1[index1,]
        index1 = index1 + 1;
    # dataset.shape

test_data_size = 50;

train_data = dataset[:-test_data_size,]
#train_data = dataset[:-test_data_size,2]
size_train_data = len(train_data)
test_data = dataset[-test_data_size:,]
#test_data = dataset[-test_data_size:,2]
print(len(train_data))
print(len(test_data))


train_data[:2,:]


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data)
#train_data_normalized = scaler.fit_transform(train_data.reshape(-1,1))

train_data_normalized[:10,:]

train_data_normalized_tensor = torch.FloatTensor(train_data_normalized).view(size_train_data,3)
#train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

train_data_normalized_tensor[:10,:]

train_window=20

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    #L = input_data.shape[0]
    for i in range(L-tw):
        train_seq = input_data[i:i+tw,:3]
        #train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw,2]
        #train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq


train_inout_seq = create_inout_sequences(train_data_normalized_tensor, train_window)

train_inout_seq[:2]

### DEEP LEARNING ARCHITECTURE

class LSTM(nn.Module):
    def __init__(self, input_size=3, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(3,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 150

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')


fut_pred = 400
size_prediction = fut_pred
#test_inputs = train_data_normalized[-train_window:].tolist()
#test_inputs = train_data_normalized[-2*train_window:-1*train_window]

test_inputs = train_inout_seq[-size_prediction:]

#print(test_inputs[0:20])


model.eval()

#train_data_normalized = torch.FloatTensor(train_data_normalized).view(size_train_data,3)
 #train_seq = input_data[i:i+tw,:3]
prediction = [];
for i in range(fut_pred):
    #we consider train_window window size to do the prediction
    seq = test_inputs[i][0]
    #seq = seq1[:train_window,:3]
    with torch.no_grad():
        model.hidden = (torch.zeros(3, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        # The prediction  vector save the prediction of label in each iteration.
        prediction.append(model(seq).item())
        #model(seq).item()
       # test_inputs.append(model(seq).item())
        #print(model(seq).item())


prediction = numpy.array(prediction)
prediction = prediction.reshape(fut_pred,1)
prediction.shape

# actual_predictions = scaler.inverse_transform(train_data_normalized[1:10,])
# print(actual_predictions)

prediction_full = numpy.column_stack([prediction, prediction, prediction])

prediction_full_tensor = torch.FloatTensor(prediction_full).view(size_prediction, 3)

actual_prediction = scaler.inverse_transform(prediction_full_tensor)

for i in range(size_prediction):
    if actual_prediction[i, 2] > 1.5:
        actual_prediction[i, 2] = 2
    else:
        actual_prediction[i, 2] = 1

prediction = actual_prediction[:, 2];
print(prediction)

# for ii in range(10):
#    train_data_normalized[ii,2] = float(prediction[ii])

# actual_predictions = scaler.inverse_transform(train_data_normalized[1:10,])
# print(actual_predictions)

# print(train_data_normalized[0:20,])


# Real Data
real_data = train_data[-size_prediction:,2]
print(real_data)


error = np.abs(real_data - prediction)
print(np.sum(error)/size_prediction)
print(np.sum(error))




