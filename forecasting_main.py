import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


#cd /home/lut/Desktop/slicing_powersystems


import numpy
data1 = numpy.loadtxt('2019-04-01_00h_UTC_PMUID01.txt')
data2 = numpy.loadtxt('2019-04-01_00h_UTC_PMUID02.txt')


size_dataset = 500;
mag_VA1 = data1[:size_dataset,6];
mav_VA1phase = data1[:size_dataset,7];
mag_IA1 = data1[:size_dataset,8];
mav_IA1phase = data1[:size_dataset,9];

mag_VA2 = data2[:size_dataset,6];
mav_VA2phase = data2[:size_dataset,7];
mag_IA2 = data2[:size_dataset,8];
mav_IA2phase = data2[:size_dataset,9];

import matplotlib.pyplot as plt
#plt.plot(mag_VA1)
#plt.plot(mag_VA2)


#plt.plot(mag_IA1)
#plt.plot(mag_IA2)

info1 = numpy.column_stack([mag_VA1,mav_VA1phase,mag_IA1, mav_IA1phase])
info2 = numpy.column_stack([mag_VA2,mav_VA2phase,mag_IA2, mav_IA2phase])

#Adding Extra columns for each message type: M1 and M2
unos = numpy.ones(info1.shape[0]);
unos = unos.reshape(info1.shape[0],1);
label_1 = unos * 1;
label_2 = unos * 2;

info1 = numpy.column_stack([info1, label_1])
info2 = numpy.column_stack([info2, label_2])

# CREATING ELASTIC DATASET
size_iteration = info1.shape[0];
dataset = info1[:size_iteration, ] * 0;
index1 = 0;
index2 = 0;
for ii in range(size_iteration):

    # TO GET A RANDOM GENERATION OF EVENT 2
    s = numpy.random.uniform(0, 1, 1)
    s = 0.8;
    if s < 0.5:
        step = 3
    elif s < 0.75:
        step = 5
    else:
        step = 6

    # Generation of elastic dataset
    trigger = numpy.remainder(ii, 10);
    if trigger > step:
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
#print(len(train_data))
#print(len(test_data))


# SCALING THE DATASET between -1 and 1

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data)
#train_data_normalized = scaler.fit_transform(train_data.reshape(-1,1))

#train_data_normalized[:10,:]

train_data_normalized_tensor = torch.FloatTensor(train_data_normalized).view(size_train_data,5)
#train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

# TRAINING WINDOW SIZE
train_window=40

# Choosing information to be used on the time series prediction
# here we define what part of the dataset is the FEATURE information
# and the LABEL or target information for the training session.

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    #L = input_data.shape[0]
    for i in range(L-tw):
        train_seq = input_data[i:i+tw,:4]
        #train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw,4]
        #train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq


train_inout_seq = create_inout_sequences(train_data_normalized_tensor, train_window)

### DEEP LEARNING ARCHITECTURE

class LSTM(nn.Module):
    def __init__(self, input_size=4, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(4,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]


model = LSTM()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 50

for i in range(epochs):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        if False:
            value_aux = labels.item()
            if value_aux == 1.0:
                print(y_pred)
                print(labels)

        single_loss.backward()
        optimizer.step()
    if single_loss.item() < 0.000001:
        print('time to pull...')
        break

    if i%1 == 0:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

#EVALUATING TRAINED MODEL
#(ToDo: use the TEST vector)
fut_pred = 20
size_prediction = fut_pred

test_inputs = train_inout_seq[-size_prediction:]
#test_inputs = train_inout_seq[:size_prediction]

model.eval()

#train_data_normalized = torch.FloatTensor(train_data_normalized).view(size_train_data,3)
 #train_seq = input_data[i:i+tw,:3]
prediction = [];
for i in range(fut_pred):
    #we consider train_window window size to do the prediction
    seq = test_inputs[i][0]
    labell= test_inputs[i][1]
    #seq = seq1[:train_window,:3]
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        # The prediction  vector save the prediction of label in each iteration.
        prediction.append(model(seq).item())
        single_loss = loss_function(model(seq), labell)
        print('++ new info ++')
        print(seq)
        print(labell)
        print(model(seq).item())
        print(single_loss)

prediction = numpy.array(prediction)
prediction = prediction.reshape(fut_pred,1)
prediction.shape

# RE-Scaling the predicted information
#
prediction_full = numpy.column_stack([prediction, prediction, prediction, prediction, prediction])

prediction_full_tensor = torch.FloatTensor(prediction_full).view(size_prediction, 5)

actual_prediction = scaler.inverse_transform(prediction_full_tensor)

actual_prediction_ref = actual_prediction

for i in range(size_prediction):
    if actual_prediction[i, 4] > 1.5:
        actual_prediction[i, 4] = 2
    else:
        actual_prediction[i, 4] = 1

prediction_end = actual_prediction[:, 4];
print(prediction_end)


# Real Data
real_data = train_data[-size_prediction:,4]
print(real_data)


error = np.abs(real_data - prediction_end)
print(np.sum(error)/size_prediction)

print(np.sum(error))
