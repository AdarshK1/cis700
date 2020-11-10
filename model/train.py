import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import wandb
from cnn_rnn import Net
from dataloaders import CIS700Dataset

# eventually we can do sweeps with this setup
hyperparameter_defaults = dict(
    batch_size=64,
    learning_rate=0.0001,
    weight_decay=0.0001,
    epochs=150
)

wandb.init(project="cis700", config=hyperparameter_defaults)
config = wandb.config

net = Net().cuda().float()

# the usual suspects
optimizer = optim.Adam(net.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                       weight_decay=config.weight_decay, amsgrad=False)

criterion = nn.MSELoss()

# instantiate the datasets
train_loaders = []
sub_dirs = [
    "/home/adarsh/ros-workspaces/cis700_workspace/src/learned_planning_pipeline/bag_harvester/cis700_data_gt/",
]

for sdir in sub_dirs:
    # no idea why it fails sometimes, here's a cheap hack
    try:
        train_loaders.append(CIS700Dataset(batch=config.batch_size, sub_dir=sdir))
    except ValueError:
        train_loaders.append(CIS700Dataset(batch=config.batch_size, sub_dir=sdir))


# train !
for epoch in range(config.epochs):
    for train_loader in train_loaders:
        # this resets which I/O pairs have been used already in the dataloader (from the last epoch)
        train_loader.reset_items_left()

        # for every batch
        for iteration in range(train_loader.num_samples // config.batch_size):
            annotated, rgb, semantic, out = train_loader.__getitem__()
            # print("annotated", annotated.shape)
            # print("rgb", rgb.shape)
            # print("semantic", semantic.shape)
            # print("out", out.shape)

            optimizer.zero_grad()  # zero the gradient buffers

            # numpy to float tensor and all that junk
            annotated_tensor = torch.from_numpy(annotated).float()
            rgb_tensor = torch.from_numpy(rgb).float()
            semantic_tensor = torch.from_numpy(semantic).float()
            out_tensor = torch.from_numpy(out).float()

            # forward! doesn't use semantic rn but we have it i guess
            output = net(rgb_tensor.cuda(), annotated_tensor.cuda())

            # hack because i like big numbers
            loss = 4 * criterion(output.cpu().float(), torch.from_numpy(out).float())

            wandb.log({'loader': train_loader.data_dir, 'loss': loss.item(), 'epoch': epoch, 'iteration': iteration})
            print({'loader': train_loader.data_dir, 'loss': loss.item(), 'epoch': epoch, 'iteration': iteration})

            # backprop
            loss.backward()
            optimizer.step()  # Does the update

            # import sys
            # sys.exit(0)

            backup_path = "models/model.ckpt"
            torch.save(net.state_dict(), backup_path)

    PATH = "models/model_{}.ckpt".format(epoch)
    torch.save(net.state_dict(), PATH)
    wandb.save(PATH)
