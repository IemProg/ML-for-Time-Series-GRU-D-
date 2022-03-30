# -*- coding: utf-8 -*-
from GRUD import *

import torch.utils.data as utils
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

def get_dataloaders(dataset, outcomes, x_mean, BATCH_SIZE, max_feats=49, test_proportion=0.2):
    train_data, test_data, train_label, test_label = train_test_split(dataset, outcomes, test_size=test_proportion, random_state=42)
    train_data, train_label = torch.Tensor(train_data), torch.Tensor(train_label)
    #valid_data, valid_label = torch.Tensor(valid_data), torch.Tensor(valid_label)
    test_data, test_label = torch.Tensor(test_data), torch.Tensor(test_label)

    train_dataset = utils.TensorDataset(train_data, train_label)
    #valid_dataset = utils.TensorDataset(valid_data, valid_label)
    test_dataset = utils.TensorDataset(test_data, test_label)

    train_dataloader = utils.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle=True)
    #valid_dataloader = utils.DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle=True)
    test_dataloader = utils.DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle=False)

    print("\t\t train_dataset: ", train_data.shape)
    print("\t\t test_data: ", test_data.shape)

    x_mean = torch.mean(train_data[:, 0], axis = 0)

    #return train_dataloader, valid_dataloader, test_dataloader, max_feats, x_mean
    return train_dataloader, test_dataloader, max_feats, x_mean

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(model, train_dataloader, valid_dataloader, batch_size, num_epochs = 2, patience = 10, min_delta = 0.00001, learning_rate = 0.001, learning_rate_decay=7):
    print('\t\t Model Structure: ', model)
    print('\t\t Start Training ... ')
    model.cuda()

    print("\t\t Number of parameters: ", count_parameters(model))
    print("\t\t Learning rate: {}, learning_rate_decay:{}".format(learning_rate, learning_rate_decay))
    if (type(model) == nn.modules.container.Sequential):
        output_last = model[-1].output_last
        print('Output type dermined by the last layer')
    else:
        output_last = model.output_last
        print('Output type dermined by the model')

    criterion = torch.nn.BCELoss()
    use_gpu = torch.cuda.is_available()

    interval = 100
    losses_train = []
    losses_valid = []
    losses_epochs_train = []
    losses_epochs_valid = []

    train_accuracies, val_accuracies = [], []
    cur_time = time.time()
    pre_time = time.time()

    # Variables for Early Stopping
    is_best_model = 0
    patient_epoch = 0
    for epoch in range(num_epochs):
        if learning_rate_decay != 0:
            # every [decay_step] epoch reduce the learning rate by half
            if  epoch % learning_rate_decay == 0:
                learning_rate = learning_rate/2
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                print('\t\t at epoch {} learning_rate is updated to {}'.format(epoch, learning_rate))

        trained_number = 0
        valid_dataloader_iter = iter(valid_dataloader)

        losses_epoch_train = []
        losses_epoch_valid = []
        acc, val_accuracy = [], []
        for data in train_dataloader:
            inputs, labels = data
            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            model.zero_grad()
            outputs = model(inputs)

            if output_last:
                loss_train = criterion(torch.squeeze(outputs), torch.squeeze(labels))
            else:
                full_labels = torch.cat((inputs[:,1:,:], labels), dim = 1)
                loss_train = criterion(outputs, full_labels)

            acc.append(torch.eq((torch.squeeze(outputs.data) > 0.5).float(), torch.squeeze(labels)))

            losses_train.append(loss_train.data.detach().cpu().numpy())
            losses_epoch_train.append(loss_train.data.detach().cpu().numpy())

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

             # validation
            with torch.no_grad():
                try:
                    inputs_val, labels_val = next(valid_dataloader_iter)
                except StopIteration:
                    valid_dataloader_iter = iter(valid_dataloader)
                    inputs_val, labels_val = next(valid_dataloader_iter)

                if use_gpu:
                    inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
                else:
                    inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

                model.zero_grad()

                outputs_val = model(inputs_val)
                if output_last:
                    loss_valid = criterion(torch.squeeze(outputs_val), torch.squeeze(labels_val))
                else:
                    full_labels_val = torch.cat((inputs_val[:,1:,:], labels_val), dim = 1)
                    loss_valid = criterion(outputs_val, full_labels_val)

                losses_valid.append(loss_valid.data.data.detach().cpu().numpy())
                losses_epoch_valid.append(loss_valid.data.data.detach().cpu().numpy())

                val_accuracy.append(torch.eq((torch.squeeze(outputs_val.data) > 0.5).float(), torch.squeeze(labels_val)))
            # output
            trained_number += 1

        train_acc = torch.mean(torch.cat(acc).float()).data.detach().cpu().numpy()
        val_acc = torch.mean(torch.cat(val_accuracy).float()).data.detach().cpu().numpy()

        avg_losses_epoch_train = sum(losses_epoch_train)/ float(len(losses_epoch_train))
        avg_losses_epoch_valid = sum(losses_epoch_valid)/ float(len(losses_epoch_valid))
        losses_epochs_train.append(avg_losses_epoch_train)
        losses_epochs_valid.append(avg_losses_epoch_valid)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        # Early Stopping
        if epoch == 0:
            is_best_model = 1
            best_model = model
            min_loss_epoch_valid = 10000.0
            if avg_losses_epoch_valid < min_loss_epoch_valid:
                min_loss_epoch_valid = avg_losses_epoch_valid
        else:
            if min_loss_epoch_valid - avg_losses_epoch_valid > min_delta:
                is_best_model = 1
                best_model = model
                min_loss_epoch_valid = avg_losses_epoch_valid
                patient_epoch = 0
            else:
                is_best_model = 0
                patient_epoch += 1
                if patient_epoch >= patience:
                    print('Early Stopped at Epoch:', epoch)
                    break

        # Print training parameters
        cur_time = time.time()
        print('Epoch: {}, train_loss: {}, valid_loss: {}, train_acc: {}, val_acc: {}, time: {}, best model: {}'.format( \
                    epoch, \
                    np.around(avg_losses_epoch_train, decimals=8),\
                    np.around(avg_losses_epoch_valid, decimals=8),\
                    np.around(train_acc, decimals=8),\
                    np.around(val_acc, decimals=8),\
                    np.around([cur_time - pre_time] , decimals=2),\
                    is_best_model))
        pre_time = cur_time

    return best_model, [losses_train, losses_valid, losses_epochs_train, losses_epochs_valid, train_accuracies, val_accuracies]

def test_model_gru_d(loader, model):
    correct, total, total_loss = 0, 0, 0
    model.eval()
    predictions, truths = [], []

    for data, label in loader:
        x, m, delta = data[:, 0], data[:, 1], data[:, 2]

        x = Variable(x.float()).cuda()
        delta = Variable(delta.float()).cuda()
        m = Variable(m.float()).cuda()

        output, hidden = gru_d(x, delta, m)

        label = label
        loss = loss_func(output, Variable(label.long()).cuda())

        total_loss += loss.item()
        predicted = (output.cpu().max(1)[1].data.long()).view(-1)
#         predicted = ((output.cpu().data > 0.5).long()).view(-1)
        predictions += list(predicted.numpy())
        truths += list(label.numpy())
        total += label.size(0)
        correct += (predicted == label).sum()

    model.train()

    return 100 * correct / total, roc_auc_score(truths, predictions), total_loss/len(loader)

if __name__ == "__main__":
    data = 'detection'
    if data == 'detection':
        speed_matrix =  np.load("dataset.npy", allow_pickle=True)
        labels = np.load("y1_out.npy", allow_pickle=True)
        xmean = np.load("x_mean_aft_nor.npy", allow_pickle=True)
        outcomes = pd.read_csv('Outcomes-a.txt')

    hidden_size = 64
    gru_dropout = 0.3
    decoder_dropout = 0.5
    output_size = 2

    loss_func = torch.nn.NLLLoss(weight=Variable(torch.FloatTensor([1/sum(outcomes['In-hospital_death'] == 0), 1/sum(outcomes['In-hospital_death'] == 1)])).cuda())
    num_epochs = 40

    #train_dataloader, valid_dataloader, test_dataloader, max, X_mean = PrepareDataset(speed_matrix, BATCH_SIZE = 64, masking = True)
    train_dataloader, valid_dataloader, test_dataloader, max_feats, X_mean = get_dataloaders(speed_matrix, labels, xmean, BATCH_SIZE = B_SIZE)
    gru_d = GRUD(NUM_FEATURES, 49, output_size, gru_dropout, decoder_dropout)
    gru_d.cuda()
    optimizer = torch.optim.Adam(gru_d.parameters(), lr=1e-3)

    gru_d_train_loss_history = []
    gru_d_test_auc_history = []
    gru_d_val_loss_history = []
    gru_d_val_acc_history = []

    num_epochs = 60

    for epoch in range(num_epochs):
        start = time()
        total_loss = 0.

        for i, (data, label) in enumerate(train_loader):
            x, m, delta = data[:, 0], data[:, 1], data[:, 2]

            x = Variable(x.float()).cuda()
            delta = Variable(delta.float()).cuda()
            m = Variable(m.float()).cuda()

            optimizer.zero_grad()

            output, hidden = gru_d(x, delta, m)

            label = label
            loss = loss_func(output, Variable(label.long()).cuda())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # report performance
        with torch.no_grad():
            val_acc, test_auc, val_loss = test_model_gru_d(test_loader, gru_d)
            gru_d_train_loss_history.append(total_loss/(i+1))
            gru_d_test_auc_history.append(test_auc)
            gru_d_val_acc_history.append(val_acc)
            gru_d_val_loss_history.append(val_loss)
            print('Fold: {8}, Epoch: [{0}/{1}], Step: [{2}/{3}], Loss: {4}, Validation Loss:{5}, Validation Acc:{6}, AUC:{7}'.format(
                epoch + 1, num_epochs, i + 1, len(train_loader), total_loss/(i+1), val_loss, val_acc, test_auc, k+1))
        print("\t\t Epoch %d time: %.4f" % (epoch, time()-start))
