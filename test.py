import time
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torch
import transforms as T
from movinets import MoViNet
from movinets.config import _C


model = MoViNet(_C.MODEL.MoViNetA0, causal=False, pretrained=True)
model.classifier[3] = torch.nn.Conv3d(2048, 51, (1, 1, 1))
start_time = time.time()

if torch.cuda.is_available():
    model.load_state_dict(torch.load("a0_hmdb51_no_causal.pth"))
    model.cuda()
else:
    model.load_state_dict(torch.load("a0_hmdb51_no_causal.pth", map_location=torch.device('cpu')))
model.eval()

torch.manual_seed(97)
num_frames = 16  # 16
clip_steps = 2
Bs_Train = 16
Bs_Test = 16

transform = transforms.Compose([

    T.ToFloatTensorInZeroOne(),
    T.Resize((200, 200)),
    T.RandomHorizontalFlip(),
    # T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    T.RandomCrop((172, 172))])
transform_test = transforms.Compose([
    T.ToFloatTensorInZeroOne(),
    T.Resize((200, 200)),
    # T.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
    T.CenterCrop((172, 172))])

hmdb51_train = torchvision.datasets.HMDB51('video_data/', 'test_train_splits/', num_frames, frame_rate=5,
                                           step_between_clips=clip_steps, fold=1, train=True,
                                           transform=transform, num_workers=2)

hmdb51_test = torchvision.datasets.HMDB51('video_data/', 'test_train_splits/', num_frames, frame_rate=5,
                                          step_between_clips=clip_steps, fold=1, train=False,
                                          transform=transform_test, num_workers=2)
train_loader = DataLoader(hmdb51_train, batch_size=Bs_Train, shuffle=True)
test_loader = DataLoader(hmdb51_test, batch_size=Bs_Test, shuffle=False)


def train_iter(model, optimz, data_load, loss_val):
    samples = len(data_load.dataset)
    model.train()
    model.cuda()
    model.clean_activation_buffers()
    optimz.zero_grad()
    for i, (data, _, target) in enumerate(data_load):
        out = F.log_softmax(model(data.cuda()), dim=1)
        loss = F.nll_loss(out, target.cuda())
        loss.backward()
        optimz.step()
        optimz.zero_grad()
        model.clean_activation_buffers()
        if i % 50 == 0:
            print('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_val.append(loss.item())


def evaluate(model, data_load, loss_val):
    model.eval()

    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0
    model.clean_activation_buffers()
    with torch.no_grad():
        for data, _, target in data_load:
            if torch.cuda.is_available():
                output = F.log_softmax(model(data.cuda()), dim=1)
                loss = F.nll_loss(output, target.cuda(), reduction='sum')
                _, pred = torch.max(output, dim=1)
                tloss += loss.item()
                csamp += pred.eq(target.cuda()).sum()
            else:
                output = F.log_softmax(model(data), dim=1)
                loss = F.nll_loss(output, target, reduction='sum')
                _, pred = torch.max(output, dim=1)
                tloss += loss.item()
                csamp += pred.eq(target).sum()
            model.clean_activation_buffers()
    aloss = tloss / samples
    loss_val.append(aloss)
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * csamp / samples) + '%)\n')


def train_iter_stream(model, optimz, data_load, loss_val, n_clips=2, n_clip_frames=8):
    """
    In causal mode with stream buffer a single video is fed to the network
    using subclips of lenght n_clip_frames.
    n_clips*n_clip_frames should be equal to the total number of frames presents
    in the video.

    n_clips : number of clips that are used
    n_clip_frames : number of frame contained in each clip
    """
    # clean the buffer of activations
    samples = len(data_load.dataset)
    model.cuda()
    model.train()
    model.clean_activation_buffers()
    optimz.zero_grad()

    for i, (data, _, target) in enumerate(data_load):
        data = data.cuda()
        target = target.cuda()
        l_batch = 0
        # backward pass for each clip
        for j in range(n_clips):
            output = F.log_softmax(model(data[:, :, (n_clip_frames) * (j):(n_clip_frames) * (j + 1)]), dim=1)
            loss = F.nll_loss(output, target)
            _, pred = torch.max(output, dim=1)
            loss = F.nll_loss(output, target) / n_clips
            loss.backward()
        l_batch += loss.item() * n_clips
        optimz.step()
        optimz.zero_grad()

        # clean the buffer of activations
        model.clean_activation_buffers()
        if i % 50 == 0:
            print('[' + '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_load)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(l_batch))
            loss_val.append(l_batch)


def evaluate_stream(model, data_load, loss_val, n_clips=2, n_clip_frames=8):
    model.eval()
    model.cuda()
    samples = len(data_load.dataset)
    csamp = 0
    tloss = 0
    with torch.no_grad():
        for data, _, target in data_load:
            data = data.cuda()
            target = target.cuda()
            model.clean_activation_buffers()
            for j in range(n_clips):
                output = F.log_softmax(model(data[:, :, (n_clip_frames) * (j):(n_clip_frames) * (j + 1)]), dim=1)
                loss = F.nll_loss(output, target)
            _, pred = torch.max(output, dim=1)
            tloss += loss.item()
            csamp += pred.eq(target).sum()

    aloss = tloss / len(data_load)
    loss_val.append(aloss)
    print('\nAverage test loss: ' + '{:.4f}'.format(aloss) +
          '  Accuracy:' + '{:5}'.format(csamp) + '/' +
          '{:5}'.format(samples) + ' (' +
          '{:4.2f}'.format(100.0 * csamp / samples) + '%)\n')


N_EPOCHS = 1

trloss_val, tsloss_val = [], []

evaluate(model, test_loader, tsloss_val)

print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')