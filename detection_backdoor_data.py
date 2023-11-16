import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from model.resnet import resnet18
from model.TNet import TNet
import torch.nn.functional as F
import numpy as np
import math
import os
import random
import setproctitle
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze
from ffcv.fields.decoders import IntDecoder, RandomResizedCropRGBImageDecoder
from data_loader_detection import data_loader
import argparse

# writer
import torch.nn.functional as F
import numpy as np
import math
from ffcv.writer import DatasetWriter
from ffcv.fields import RGBImageField, IntField, TorchTensorField
import os
import setproctitle
from typing import List
import argparse
import random
from sample import get_Sample
from ffcv.transforms.common import Squeeze

proc_name = 'lover'
setproctitle.setproctitle(proc_name)


def get_acc(outputs, labels):
    """calculate acc"""
    _, predict = torch.max(outputs.data, 1)
    total_num = labels.shape[0] * 1.0
    correct_num = (labels == predict).sum().item()
    acc = correct_num / total_num
    return acc


# train one epoch
def train_loop(dataloader, model, loss_fn, optimizer):
    size, num_batches = dataloader.batch_size, len(dataloader)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    epoch_acc, epoch_loss = 0.0, 0.0
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        optimizer.zero_grad()
        pred = model(X)
        # print(pred.shape)
        # print(y.shape)

        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        epoch_acc += get_acc(pred, y)
        epoch_loss += loss.data
    print('Train loss: %.4f, Train acc: %.2f' % (epoch_loss/size, 100 * (epoch_acc / num_batches)))

def test_loop(dataloader, model, loss_fn):
    # Set the models to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = dataloader.batch_size
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the models with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            # print(pred.shape)
            # print(pred[0])
            # print(y.shape)
            # print(y[0])
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def compute_DV(T, Y, Z_, t):
    ema_rate = 0.01
    e_t2_ema = None
    t2 = T(Y, Z_)
    e_t2 = t2.exp()
    # e_t2 = e_t2.clamp(max=1e20)
    e_t2_mean = e_t2.mean()
    if e_t2_ema is None:
        loss = -(t.mean() - e_t2_mean.log())
        e_t2_ema = e_t2_mean
    else:
        """
        log(e_t2_mean)' = 1/e_t2_mean * e_t2_mean'
        e_t2_mean' = sum(e_t2')/b
        e_t2' = e_t2 * t2'
        """
        e_t2_ema = (1 - ema_rate) * e_t2_ema + ema_rate * e_t2_mean
        loss = -(t.mean() - (t2 * e_t2.detach()).mean() / e_t2_ema.item())
        # loss = -(t.mean() - e_t2_mean / e_t2_ema.item())
    return t2, e_t2, loss


def compute_infoNCE(T, Y, Z, t):
    Y_ = Y.repeat_interleave(Y.shape[0], dim=0)
    Z_ = Z.tile(Z.shape[0], 1)
    t2 = T(Y_, Z_).view(Y.shape[0], Y.shape[0], -1)
    t2 = t2.exp().mean(dim=1).log()  # mean over j
    assert t.shape == t2.shape
    loss = -(t.mean() - t2.mean())
    return t2, loss


def compute_JSD(T, Y, Z_, t):
    t2 = T(Y, Z_)
    log_t = t.sigmoid().log()
    log_t2 = (1 - t2.sigmoid()).log()
    loss = -(log_t.mean() + log_t2.mean())
    return t2, log_t, log_t2, loss

def estimate_mi(model, flag, train_loader, EPOCHS=50, mode='DV'):
  LR = 1e-6
  # train T net
  model.eval()
  (Y_dim, Z_dim) = (512, 3072) if flag == 'inputs-vs-outputs' else (10, 512)
  T = TNet(in_dim=Y_dim + Z_dim, hidden_dim=512).to(device)
  optimizer = torch.optim.Adam(T.parameters(), lr=LR, weight_decay=1e-5)
  M = []
  for t in range(EPOCHS):
    print(f"------------------------------- MI-Esti-Epoch {t + 1}-{mode} -------------------------------")
    A = []
    B = []
    L = []
    for batch, (X, _Y) in enumerate(train_loader):
      X, _Y = X.to(device), _Y.to(device)
      with torch.no_grad():
        Y = F.one_hot(_Y, num_classes=10)
        inputs = model.get_last_conv_inputs(X)
        outputs = model.get_last_conv_outputs(X)
        Y_predicted = model(X)
      if flag == 'inputs-vs-outputs':
        X = torch.flatten(X, start_dim=1)
        Y, Z_, Z = outputs, X[torch.randperm(X.shape[0])], X
      elif flag == 'Y-vs-outputs':
        Y, Z_, Z = Y_predicted, outputs[torch.randperm(outputs.shape[0])], outputs
      else:
        raise ValueError('Not supported!')
      t = T(Y, Z)
      A.append(t)
      if mode == 'DV':
        t2, e_t2, loss = compute_DV(T, Y, Z_, t)
        B.append(e_t2)
      elif mode == 'infoNCE':
        t2, loss = compute_infoNCE(T, Y, Z, t)
        B.append(t2)
      if math.isnan(loss.item()) or math.isinf(loss.item()):
        print(loss.item(), torch.isnan(t).sum(), torch.isnan(t2).sum())
        last_element = B[-1]
        B.append(last_element)
        # return M
      optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(T.parameters(), 20)
      optimizer.step()
      L.append(loss.item())
    print(f'[{mode}] loss:', np.mean(L), max(L), min(L))
    A = torch.cat(A, dim=0)
    B = torch.cat(B, dim=0)
    if mode == 'DV':
      mi = (A.mean() - B.mean().log())
    else:
      # mi = (A - B.exp().sum().log()).mean()
      mi = (A.mean() - B.mean())
    M.append(mi.item())
    print(f'[{mode}] mi:', mi.item())
  return M

def train(args, metrics, train_data_beton_path, test_data_beton_path, mode='DV'):
    """ flag = inputs-vs-outputs or Y-vs-outputs """
    batch_size = 256
    learning_rate = 1e-5

    # Data decoding and augmentation
    image_pipeline = [ToTensor(), ToDevice(device)]
    label_pipeline = [IntDecoder(), ToTensor(), ToDevice(device), Squeeze()]

    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    }
    num_workers = 6

    train_dataloader_path = train_data_beton_path
    train_dataloader = Loader(train_dataloader_path, batch_size=batch_size, num_workers=num_workers,
                              order=OrderOption.RANDOM, pipelines=pipelines)

    test_dataloader_path = test_data_beton_path
    test_dataloader = Loader(test_dataloader_path, batch_size=batch_size, num_workers=num_workers,
                             order=OrderOption.RANDOM, pipelines=pipelines)

    #model = resnet18(num_classes=10)
    #model.to(device)
    #model.train()

    #loss_fn = nn.CrossEntropyLoss()
    #optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    epochs = 100
    mi_total_metrics = []
    # detect all the classes
    for observe_class in range(1,10):
        class_time_turning_point = -1
        print(f"Start Recording Class: {observe_class} Metrics")
        mi_class_metrics = []
        for times in range(20):
            model = resnet18(num_classes=10)
            model.to(device)
            model.train()
            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
            mi_class_metrics_single_time = []
            print(f"Start Recording Class: {observe_class} Metrics Times: {times}")
            cur_dir = os.path.dirname(__file__)
            observe_class_path = os.path.join(cur_dir, f"{args.attack_type}_observe_data_class_{observe_class}.beton")
            train_dataloader_label1_path = observe_class_path
            train_dataloader_label1 = Loader(train_dataloader_label1_path, batch_size=batch_size, num_workers=num_workers,
                                             order=OrderOption.RANDOM, pipelines=pipelines)
            observe_window_x = []
            observe_window_y = []
            conv_window_x = []
            conv_window_y = []
            MI_X = []
            MI_Y = []

            for t in range(1, epochs):
                print(f"------------------------------- ResNet18 Training Epoch {t} -------------------------------")
                train_loop(train_dataloader, model, loss_fn, optimizer)
                test_loop(test_dataloader, model, loss_fn)
                if class_time_turning_point != -1:
                    if t == 1:
                        print("========== Observe MI(X,T)==============")
                        MI_X.append(estimate_mi(model, 'inputs-vs-outputs', train_dataloader_label1, EPOCHS=200, mode=mode))
                        print("========== Observe MI(Y,T)==============")
                        MI_Y.append(estimate_mi(model, 'Y-vs-outputs', train_dataloader_label1, EPOCHS=200, mode=mode))
                        mi_class_metrics_single_time.append(np.mean(MI_X[0][-5:]))
                        mi_class_metrics_single_time.append(np.mean(MI_Y[0][-5:]))
                        print("initial_point: ", mi_class_metrics_single_time)
                    if t == class_time_turning_point:
                        if class_time_turning_point == 1:
                            mi_class_metrics_single_time.append(np.mean(MI_X[0][-5:]))
                            mi_class_metrics_single_time.append(np.mean(MI_Y[0][-5:]))
                            mi_class_metrics_single_time.append(1)
                            print("initial_point: ", mi_class_metrics_single_time)
                        else:
                            turnining_x = estimate_mi(model, 'inputs-vs-outputs', train_dataloader_label1, EPOCHS=200, mode=mode)
                            turning_y = estimate_mi(model, 'Y-vs-outputs', train_dataloader_label1, EPOCHS=200, mode=mode)
                            mi_class_metrics_single_time.append(np.mean(turnining_x[-5:]))
                            mi_class_metrics_single_time.append(np.mean(turning_y[-5:]))
                            mi_class_metrics_single_time.append(class_time_turning_point)
                            print("turining point: ", mi_class_metrics_single_time)
                if class_time_turning_point == -1 and t <= 4:
                    print("========== Observe MI(X,T)==============")
                    MI_X.append(estimate_mi(model, 'inputs-vs-outputs', train_dataloader_label1, EPOCHS=200, mode=mode))
                    print("========== Observe MI(Y,T)==============")
                    MI_Y.append(estimate_mi(model, 'Y-vs-outputs', train_dataloader_label1, EPOCHS=200, mode=mode))
                    # record the initial value
                    if t == 1:
                        mi_class_metrics_single_time.append(np.mean(MI_X[0][-5:]))
                        mi_class_metrics_single_time.append(np.mean(MI_Y[0][-5:]))
                        print("initial_point: ", mi_class_metrics_single_time)
                    if t == 4:
                        first_point = MI_X[0][-5:]
                        second_point = MI_X[1][-5:]
                        third_point = MI_X[2][-5:]
                        fourth_point = MI_X[3][-5:]
                        # first point is turning point
                        if first_point > second_point > third_point > fourth_point:
                            mi_class_metrics_single_time.append(np.mean(MI_X[0][-5:]))
                            mi_class_metrics_single_time.append(np.mean(MI_Y[0][-5:]))
                            mi_class_metrics_single_time.append(1)
                            class_time_turning_point = 1
                            print("turning point is initial point")
                        elif first_point < second_point > third_point > fourth_point:
                            mi_class_metrics_single_time.append(np.mean(MI_X[1][-5:]))
                            mi_class_metrics_single_time.append(np.mean(MI_Y[1][-5:]))
                            mi_class_metrics_single_time.append(2)
                            class_time_turning_point = 2
                            print("turning point epoch is: ", 2)
                        elif first_point < second_point < third_point > fourth_point:
                            mi_class_metrics_single_time.append(np.mean(MI_X[2][-5:]))
                            mi_class_metrics_single_time.append(np.mean(MI_Y[2][-5:]))
                            mi_class_metrics_single_time.append(3)
                            class_time_turning_point = 3
                            print("turning point epoch is: ", 3)
                        elif first_point < second_point < third_point < fourth_point:
                            print("Continue finding turning point")
                if class_time_turning_point == -1 and (5 <= t < 98):
                     if t == 5:
                        observe_window_x.append(estimate_mi(model, 'inputs-vs-outputs', train_dataloader_label1, EPOCHS=200, mode=mode))
                        observe_window_y.append(estimate_mi(model, 'Y-vs-outputs', train_dataloader_label1, EPOCHS=200, mode=mode))
                        if observe_window_x[0][-5:] < MI_X[3][-5:]:
                            mi_class_metrics_single_time.append(np.mean(MI_X[3][-5:]))
                            mi_class_metrics_single_time.append(np.mean(MI_Y[3][-5:]))
                            mi_class_metrics_single_time.append(4)
                            class_time_turning_point = 4
                            print("find the turning point ", mi_class_metrics_single_time)
                     if class_time_turning_point != -1 and t == 10:
                        observe_window_x.append(estimate_mi(model, 'inputs-vs-outputs', train_dataloader_label1, EPOCHS=200, mode=mode))
                        observe_window_y.append(estimate_mi(model, 'Y-vs-outputs', train_dataloader_label1, EPOCHS=200, mode=mode))
                        if observe_window_x[2][-5:] < observe_window_x[1][-5:]:
                            mi_class_metrics_single_time.append(np.mean(observe_window_x[1][-5:]))
                            mi_class_metrics_single_time.append(np.mean(observe_window_y[1][-5:]))
                            mi_class_metrics_single_time.append(5)
                            class_time_turning_point = 5
                            print("find the turning point ", mi_class_metrics_single_time)
                     if class_time_turning_point != -1 and t == 15:
                        observe_window_x.append(estimate_mi(model, 'inputs-vs-outputs', train_dataloader_label1, EPOCHS=200, mode=mode))
                        observe_window_y.append(estimate_mi(model, 'Y-vs-outputs', train_dataloader_label1, EPOCHS=200, mode=mode))
                        if observe_window_x[3][-5:] < observe_window_x[2][-5:]:
                            mi_class_metrics_single_time.append(np.mean(observe_window_x[2][-5:]))
                            mi_class_metrics_single_time.append(np.mean(observe_window_y[2][-5:]))
                            mi_class_metrics_single_time.append(10)
                            class_time_turning_point = 10
                            print("find the turning point ", mi_class_metrics_single_time)
                     if class_time_turning_point != -1 and t == 20:
                        observe_window_x.append(estimate_mi(model, 'inputs-vs-outputs', train_dataloader_label1, EPOCHS=200, mode=mode))
                        observe_window_y.append(estimate_mi(model, 'Y-vs-outputs', train_dataloader_label1, EPOCHS=200, mode=mode))
                        if observe_window_x[3][-5:] < observe_window_x[2][-5:]:
                            mi_class_metrics_single_time.append(np.mean(observe_window_x[2][-5:]))
                            mi_class_metrics_single_time.append(np.mean(observe_window_y[2][-5:]))
                            mi_class_metrics_single_time.append(15)
                            class_time_turning_point = 15
                            print("find the turning point ", mi_class_metrics_single_time)
                     if class_time_turning_point != -1 and t == 25:
                        observe_window_x.append(estimate_mi(model, 'inputs-vs-outputs', train_dataloader_label1, EPOCHS=200, mode=mode))
                        observe_window_y.append(estimate_mi(model, 'Y-vs-outputs', train_dataloader_label1, EPOCHS=200, mode=mode))
                        if observe_window_x[4][-5:] < observe_window_x[3][-5:]:
                            mi_class_metrics_single_time.append(np.mean(observe_window_x[3][-5:]))
                            mi_class_metrics_single_time.append(np.mean(observe_window_y[3][-5:]))
                            mi_class_metrics_single_time.append(20)
                            class_time_turning_point = 20
                            print("find the turning point ", mi_class_metrics_single_time)
                     if class_time_turning_point != -1 and t == 30:
                        observe_window_x.append(estimate_mi(model, 'inputs-vs-outputs', train_dataloader_label1, EPOCHS=200, mode=mode))
                        observe_window_y.append(estimate_mi(model, 'Y-vs-outputs', train_dataloader_label1, EPOCHS=200, mode=mode))
                        if observe_window_x[5][-5:] < observe_window_x[4][-5:]:
                            mi_class_metrics_single_time.append(np.mean(observe_window_x[4][-5:]))
                            mi_class_metrics_single_time.append(np.mean(observe_window_y[4][-5:]))
                            mi_class_metrics_single_time.append(25)
                            class_time_turning_point = 25
                            print("find the turning point ", mi_class_metrics_single_time)
                if epochs - t <= 2:
                    print("========== Observe MI(X,T)==============")
                    mi_x = estimate_mi(model, 'inputs-vs-outputs', train_dataloader_label1, EPOCHS=200, mode=mode)
                    print("========== Observe MI(Y,T)==============")
                    mi_y = estimate_mi(model, 'Y-vs-outputs', train_dataloader_label1, EPOCHS=200, mode=mode)
                    conv_window_x.append(np.mean(mi_x[-5:]))
                    conv_window_y.append(np.mean(mi_y[-5:]))
                    if t == epochs -1:
                        mi_class_metrics_single_time.append(np.mean(conv_window_x))
                        mi_class_metrics_single_time.append(np.mean(conv_window_y))
                        mi_class_metrics_single_time.append(observe_class)
                        mi_class_metrics.append(mi_class_metrics_single_time)
                        print("conv point ", mi_class_metrics_single_time)
           # torch.save(model, 'models.pth')
            mi_class_metrics.append(metrics)
            print('mi_class_metrics ', mi_class_metrics)
        mi_total_metrics.append(mi_class_metrics)
        print('mi_total_metrics ', mi_total_metrics)
    return MI_X, MI_Y, mi_total_metrics


def data_loader(args):
    print("Writing the dataset as .beton file")
    device = 'cpu'
    training_data_npy = np.load(args.train_data_path)
    test_data_npy = np.load(args.test_data_path)

    train_dataset = TensorDataset(
        torch.tensor(training_data_npy['arr_0'], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        torch.tensor(training_data_npy['arr_1'], dtype=torch.long, device=device))
    test_dataset = TensorDataset(
        torch.tensor(test_data_npy['arr_0'], dtype=torch.float32, device=device).permute(0, 3, 1, 2),
        torch.tensor(test_data_npy['arr_1'], dtype=torch.long, device=device))

    from ffcv.writer import DatasetWriter
    from ffcv.fields import RGBImageField, IntField

    cur_dir = os.path.dirname(__file__)
    train_write_path = os.path.join(cur_dir, f"{args.attack_type}_train_data.beton")
    test_write_path = os.path.join(cur_dir, f"{args.attack_type}_test_data.beton")
    # Pass a type for each data field
    train_writer = DatasetWriter(train_write_path, {
        # Tune options to optimize dataset size, throughput at train-time
        'image': TorchTensorField(dtype=torch.float32, shape=(3, 32, 32)),
        'label': IntField()
    })

    # Pass a type for each data field
    test_writer = DatasetWriter(test_write_path, {
        # Tune options to optimize dataset size, throughput at train-time
        'image': TorchTensorField(dtype=torch.float32, shape=(3, 32, 32)),
        'label': IntField()
    })

    # Write dataset
    # writer.from_indexed_dataset(sample_dataset)
    print("===========Write Train Data as .beton file============")
    train_writer.from_indexed_dataset(train_dataset)
    print("===========Write Test Data as .beton file=============")
    test_writer.from_indexed_dataset(test_dataset)

    for i in range(10):
        cur_dir = os.path.dirname(__file__)
        sample_write_path = os.path.join(cur_dir, f"{args.attack_type}_observe_data_class_{i}.beton")
        # Pass a type for each data field
        sample_writer = DatasetWriter(sample_write_path, {
            # Tune options to optimize dataset size, throughput at train-time
            'image': TorchTensorField(dtype=torch.float32, shape=(3, 32, 32)),
            'label': IntField()
        })
        observe_data_npy = training_data_npy['arr_0'][training_data_npy['arr_1'] == i]
        observe_label_npy = training_data_npy['arr_1'][training_data_npy['arr_1'] == i]

        observe_data = TensorDataset(
            torch.tensor(observe_data_npy, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
            torch.tensor(observe_label_npy, dtype=torch.long, device=device))

        data_label_pairs = list(zip(observe_data_npy, observe_label_npy))
        random.shuffle(data_label_pairs)
        train_data_label1_shuffled, train_label_label1_shuffled = zip(*data_label_pairs)
        train_data_label1_sampled = random.sample(train_data_label1_shuffled, args.sampling_datasize)
        train_label_label1_sampled = np.array(random.sample(train_label_label1_shuffled, args.sampling_datasize))
        train_data_label1_sampled = np.array(train_data_label1_sampled)
        if i == 0:
            image_shuffle, label_shuffle = get_Sample(args.sampling_datasize, args.train_cleandata_path,
                                                      args.train_poisondata_path)
            sample_dataset = TensorDataset(
                torch.tensor(image_shuffle, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
                torch.tensor(label_shuffle, dtype=torch.long, device=device))
        else:
            sample_dataset = TensorDataset(
                torch.tensor(train_data_label1_sampled, dtype=torch.float32, device=device).permute(0, 3, 1, 2),
                torch.tensor(train_label_label1_sampled, dtype=torch.long, device=device))
        print(f"=======Write sample data class={i} as .beton file=========")
        sample_writer.from_indexed_dataset(sample_dataset)
    return train_write_path, test_write_path


def ob_infoNCE(args, train_data_beton_path, test_data_beton_path):
    outputs_dir = args.outputs_dir
    metrics = []
    infoNCE_MI_log_inputs_vs_outputs, infoNCE_MI_log_Y_vs_outputs, metrics = train(args, metrics, train_data_beton_path, test_data_beton_path, 'infoNCE')
    print("metrics ", metrics)
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    np.save(f'{outputs_dir}/infoNCE_MI_I(X,T).npy', infoNCE_MI_log_inputs_vs_outputs)
    np.save(f'{outputs_dir}/infoNCE_MI_I(Y,T).npy', infoNCE_MI_log_Y_vs_outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputs_dir', type=str, default='results/ob_infoNCE_06_22', help='output_dir')
    parser.add_argument('--sampling_datasize', type=int, default='4000', help='sampling_datasize')
    parser.add_argument('--training_epochs', type=str, default='100', help='training_epochs')
    parser.add_argument('--batch_size', type=str, default='256', help='batch_size')
    parser.add_argument('--learning_rate', type=str, default='1e-5', help='learning_rate')
    parser.add_argument('--mi_estimate_epochs', type=str, default='300', help='mi_estimate_epochs')
    parser.add_argument('--mi_estimate_lr', type=str, default='1e-6', help='mi_estimate_lr')
    parser.add_argument('--class', type=str, default='0', help='class')
    parser.add_argument('--attack_type', type=str, default='badNet', help='attack_type')
    parser.add_argument('--train_data_path', type=str, default='0', help='class')
    parser.add_argument('--test_data_path', type=str, default='0', help='class')
    parser.add_argument('--sample_data_path', type=str, default='0', help='class')
    parser.add_argument('--train_cleandata_path', type=str, default='data/clean_0.9.npz',
                        help='Path to the training clean data')
    parser.add_argument('--train_poisondata_path', type=str, default='data/poison_0.1.npz',
                        help='Path to the training poison data')
    args = parser.parse_args()
    train_data_beton_path, test_data_beton_path = data_loader(args)
    device = torch.device('cuda')
    # ob_DV()
    ob_infoNCE(args, train_data_beton_path, test_data_beton_path)