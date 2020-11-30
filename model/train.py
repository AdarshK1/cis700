import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import wandb
from cnn_rnn import Net
from dataloaders_v2 import CIS700Dataset
from torch.utils.data import DataLoader
import time
import sys
import cv2
import pickle
import random

# eventually we can do sweeps with this setup
hyperparameter_defaults = dict(
    batch_size=96,
    learning_rate=0.001,
    weight_decay=0.0005,
    epochs=10,
    test_iters=50,
    num_workers=48,
    map_size=70,
    loaders_from_scratch=True,
    test_only=False
)

wandb.init(project="cis700", config=hyperparameter_defaults)
config = wandb.config

net = Net().cuda().float()

# the usual suspects
optimizer = optim.Adam(net.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                       weight_decay=config.weight_decay, amsgrad=False)

criterion = nn.L1Loss(reduction="sum")

test_filename_stub = "imgs/epoch_{}_{}.png"
train_filename_stub = "imgs/epoch_{}_{}_{}.png"

train_loaders = []
test_loaders = []


if config.loaders_from_scratch:
    # instantiate the datasets
    train_sub_dirs = [
        "/home/adarsh/HDD1/cis700_final/processed/20201123-191213/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201123-203837/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201123-194327/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201123-191455/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201123-195338/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201124-034641/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201123-191855/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201123-203414/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201124-034755/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201124-034541/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201123-194129/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201123-193111/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201123-190511/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201123-190958/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201123-190928/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201123-184629/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201123-195448/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201123-194224/",
    ]

    test_sub_dirs = [
        "/home/adarsh/HDD1/cis700_final/processed/20201124-034255/",
    ]

    config_file = "/home/adarsh/ros-workspaces/cis700_workspace/src/rosbag-dl-utils/harvester_configs/cis700.yaml"

    for sdir in train_sub_dirs:
        # no idea why it fails sometimes, here's a cheap hack
        try:
            train_loaders.append(DataLoader(CIS700Dataset(config_file, sdir, map_size=config.map_size),
                                            batch_size=config.batch_size,
                                            num_workers=config.num_workers, shuffle=True))
        except ValueError:
            train_loaders.append(DataLoader(CIS700Dataset(config_file, sdir, map_size=config.map_size),
                                            batch_size=config.batch_size,
                                            num_workers=config.num_workers, shuffle=True))
        # pickle.dump(train_loaders, open("train_loaders.pkl", 'wb'))

    for sdir in test_sub_dirs:
        # no idea why it fails sometimes, here's a cheap hack
        try:
            test_loaders.append(DataLoader(CIS700Dataset(config_file, sdir, map_size=config.map_size),
                                           batch_size=1, shuffle=True))
        except ValueError:
            test_loaders.append(DataLoader(CIS700Dataset(config_file, sdir, map_size=config.map_size),
                                           batch_size=1, shuffle=True))

        # pickle.dump(test_loaders, open("test_loaders.pkl", 'wb'))
else:
    train_loaders = pickle.load(open("train_loaders.pkl", 'rb'))
    random.shuffle(train_loaders)

    test_loaders = pickle.load(open("test_loaders.pkl", 'rb'))
    random.shuffle(test_loaders)


if config.test_only:
    net.load_state_dict(
        torch.load(
            "/home/adarsh/ros-workspaces/cis700_workspace/src/cis700/model/models/model_0.ckpt"))

# let's do and save some viz stuff
def torch_to_cv2(out):
    # print(np.min(out), np.max(out))
    out = out.numpy()[0, :, :, :]
    out -= np.min(out)
    out *= 255.0 / (np.max(out))
    out = np.moveaxis(out, 0, 2)
    out = np.append(out, np.zeros((out.shape[0], out.shape[1], 1)), axis=2)
    return out


# train !
for epoch in range(config.epochs):
    if not config.test_only:
        random.shuffle(train_loaders)
        for train_loader in train_loaders:
            for i_batch, sample_batched in enumerate(train_loader):
                t1 = time.time()
                annotated, rgb, semantic, out = sample_batched

                optimizer.zero_grad()  # zero the gradient buffers

                # numpy to float tensor and all that junk
                annotated_tensor = annotated.float()
                rgb_tensor = rgb.float()
                semantic_tensor = semantic.float()
                out_tensor = out.float()

                # forward! doesn't use semantic rn but we have it i guess
                output = net(rgb_tensor.cuda(), annotated_tensor.cuda())

                # hack because i like big numbers
                loss = criterion(output.cpu().float(), out_tensor)

                wandb.log({'epoch': epoch, 'iteration': i_batch, 'loss': loss.item(), 'loader': train_loader.dataset.data_dir})
                print({'epoch': epoch, 'iteration': i_batch, 'loss': loss.item(), 'loader': train_loader.dataset.data_dir})

                # backprop
                loss.backward()
                optimizer.step()  # Does the update

                backup_path = "models_v4/model.ckpt"
                torch.save(net.state_dict(), backup_path)
                t2 = time.time()

                if i_batch % 10 == 0:
                    out_gt = torch_to_cv2(out)
                    cv2.imwrite(train_filename_stub.format(epoch, i_batch, "train_out_gt"), out_gt)
                    wandb.log({"train_out_gt": [wandb.Image(out_gt, caption=str(epoch))]})

                    annotated_disp = torch_to_cv2(annotated)
                    cv2.imwrite(train_filename_stub.format(epoch, i_batch, "train_annotated"), annotated_disp)
                    wandb.log({"train_annotated": [wandb.Image(annotated, caption=str(epoch))]})

                    out_pred = torch_to_cv2(output.cpu().detach().float())
                    cv2.imwrite(train_filename_stub.format(epoch, i_batch, "train_out_pred"), out_pred)
                    wandb.log({"train_out_pred": [wandb.Image(out_pred, caption=str(epoch))]})

                    cv2.imshow("out_gt", out_gt)
                    cv2.imshow("annotated", annotated_disp)
                    cv2.imshow("out_pred", out_pred)
                    cv2.waitKey(1000)

    random.shuffle(test_loaders)
    for test_loader in test_loaders:
        losses = 0
        for i_batch, sample_batched in enumerate(test_loader):
            if i_batch > config.test_iters:
                break

            annotated, rgb, semantic, out = sample_batched

            # numpy to float tensor and all that junk
            annotated_tensor = annotated.float()
            rgb_tensor = rgb.float()
            semantic_tensor = semantic.float()
            out_tensor = out.float()

            # forward! doesn't use semantic rn but we have it i guess
            output = net(rgb_tensor.cuda(), annotated_tensor.cuda())

            # hack because i like big numbers
            loss = criterion(output.cpu().float(), out_tensor)
            losses += loss.item()

            out_gt = torch_to_cv2(out)
            cv2.imwrite(test_filename_stub.format(epoch, "test_out_gt"), out_gt)
            wandb.log({"test_out_gt": [wandb.Image(out_gt, caption=str(epoch))]})

            annotated_disp = torch_to_cv2(annotated)
            cv2.imwrite(test_filename_stub.format(epoch, "test_annotated"), annotated_disp)
            wandb.log({"test_annotated": [wandb.Image(annotated, caption=str(epoch))]})

            out_pred = torch_to_cv2(output.cpu().detach().float())
            cv2.imwrite(test_filename_stub.format(epoch, "test_out_pred"), out_pred)
            wandb.log({"test_out_pred": [wandb.Image(out_pred, caption=str(epoch))]})

            wandb.log({'test_loss': losses / config.test_iters})
            print({'test_loss': loss})

            # if i_batch == 0:
            cv2.imshow("out_gt", out_gt)
            cv2.imshow("annotated", annotated_disp)
            cv2.imshow("out_pred", out_pred)
            cv2.waitKey(1000)

    PATH = "models/model_{}.ckpt".format(epoch)
    torch.save(net.state_dict(), PATH)
    wandb.save(PATH)
