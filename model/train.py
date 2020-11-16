import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import wandb
from cnn_rnn import Net
from dataloaders import CIS700Dataset
from torch.utils.data import DataLoader
import time
import sys
import cv2

# eventually we can do sweeps with this setup
hyperparameter_defaults = dict(
    batch_size=32,
    learning_rate=0.0001,
    weight_decay=0.00005,
    epochs=150,
    test_iters=15,
    num_workers=16,
    map_size=70
)

wandb.init(project="cis700", config=hyperparameter_defaults)
config = wandb.config

net = Net().cuda().float()

# the usual suspects
optimizer = optim.Adam(net.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                       weight_decay=config.weight_decay, amsgrad=False)

criterion = nn.MSELoss()

test_filename_stub = "imgs/epoch_{}_{}.png"

# instantiate the datasets
train_loaders = []
train_sub_dirs = [
    "/home/adarsh/ros-workspaces/cis700_workspace/src/learned_planning_pipeline/bag_harvester/cis700_data_gt/",
]

test_loaders = []
test_sub_dirs = [
    "/home/adarsh/ros-workspaces/cis700_workspace/src/learned_planning_pipeline/bag_harvester/cis700_data_gt/",
]

for sdir in train_sub_dirs:
    # no idea why it fails sometimes, here's a cheap hack
    try:
        train_loaders.append(DataLoader(CIS700Dataset(batch=1, sub_dir=sdir, map_size=config.map_size),
                                        batch_size=config.batch_size,
                                        num_workers=config.num_workers))
    except ValueError:
        train_loaders.append(DataLoader(CIS700Dataset(batch=1, sub_dir=sdir, map_size=config.map_size),
                                        batch_size=config.batch_size,
                                        num_workers=config.num_workers))

for sdir in test_sub_dirs:
    # no idea why it fails sometimes, here's a cheap hack
    try:
        test_loaders.append(DataLoader(CIS700Dataset(batch=1, sub_dir=sdir, map_size=config.map_size),
                                       batch_size=1))
    except ValueError:
        test_loaders.append(DataLoader(CIS700Dataset(batch=1, sub_dir=sdir, map_size=config.map_size),
                                       batch_size=1))

# train !
for epoch in range(config.epochs):
    for train_loader in train_loaders:
        # this resets which I/O pairs have been used already in the dataloader (from the last epoch)
        train_loader.dataset.reset_items_left()

        # for every batch
        # for iteration in range(len(train_loader.dataset) // config.batch_size):
        for i_batch, sample_batched in enumerate(train_loader):
            t1 = time.time()
            annotated, rgb, semantic, out = sample_batched

            # print("annotated shape:", annotated.shape)
            # print("rgb shape:", rgb.shape)
            # print("semantic shape:", semantic.shape)
            # print("out shape:", out.shape)

            optimizer.zero_grad()  # zero the gradient buffers

            # numpy to float tensor and all that junk
            annotated_tensor = annotated.float()
            rgb_tensor = rgb.float()
            semantic_tensor = semantic.float()
            out_tensor = out.float()

            # forward! doesn't use semantic rn but we have it i guess
            output = net(rgb_tensor.cuda(), annotated_tensor.cuda())

            # hack because i like big numbers
            loss = 4 * criterion(output.cpu().float(), out_tensor)

            wandb.log({'epoch': epoch, 'iteration': i_batch, 'loss': loss.item(), 'loader': train_loader.dataset.data_dir})
            print({'epoch': epoch, 'iteration': i_batch, 'loss': loss.item(), 'loader': train_loader.dataset.data_dir})

            # backprop
            loss.backward()
            optimizer.step()  # Does the update

            backup_path = "models/model.ckpt"
            torch.save(net.state_dict(), backup_path)
            t2 = time.time()
            # print("dataload time", t2 - t1)

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
            loss = 4 * criterion(output.cpu().float(), out_tensor)
            losses += loss.item()

            # let's do and save some viz stuff
            def torch_to_cv2(out):
                out = out.numpy()[0, :, :, :]
                out -= np.min(out)
                out *= 255.0 / (np.max(out))
                out = np.moveaxis(out, 0, 2)
                return out

            out_gt = torch_to_cv2(out)
            cv2.imwrite(test_filename_stub.format(epoch, "out_gt"), out_gt)
            wandb.log({"out_gt": [wandb.Image(out_gt, caption=str(epoch))]})

            annotated_disp = torch_to_cv2(annotated)
            cv2.imwrite(test_filename_stub.format(epoch, "annotated"), annotated_disp)
            wandb.log({"annotated": [wandb.Image(annotated, caption=str(epoch))]})

            out_pred = torch_to_cv2(output.cpu().detach().float())
            cv2.imwrite(test_filename_stub.format(epoch, "out_pred"), out_pred)
            wandb.log({"out_pred": [wandb.Image(out_pred, caption=str(epoch))]})

            if i_batch == 0:
                cv2.imshow("out_gt", out_gt)
                cv2.imshow("annotated", annotated_disp)
                cv2.imshow("out_pred", out_pred)
                cv2.waitKey(10000)

        wandb.log({'test_loss': losses / config.test_iters})
        print({'test_loss': losses / config.test_iters})

    PATH = "models/model_{}.ckpt".format(epoch)
    torch.save(net.state_dict(), PATH)
    wandb.save(PATH)
