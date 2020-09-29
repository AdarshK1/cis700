import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
# import wandb

# wandb.init(project="cis700")



class Net(nn.module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1_occ_branch = nn.Conv2d(3, 10, kernel_size=3)
        self.conv2_occ_branch = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3_occ_branch = nn.Conv2d(20, 40, kernel_size=3)
        self.conv4_occ_branch = nn.Conv2d(40, 80, kernel_size=3)

        self.conv1_rgb_branch = nn.Conv2d(3, 10, kernel_size=3)
        self.conv2_rgb_branch = nn.Conv2d(10, 20, kernel_size=3)
        self.conv3_rgb_branch = nn.Conv2d(20, 40, kernel_size=3)
        self.conv4_rgb_branch = nn.Conv2d(40, 80, kernel_size=3)

        self.fc1 = nn.Linear(500, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 20)

    '''
    rgb is just an rgb image, occ_plus is a 3 channel data structure with the occupancy grid, suggested path and 
    goal all encoded in the different channels
    '''
    def forward(self, rgb, occ_plus):
        rgb_branch = F.relu(F.max_pool2d(self.conv1_rgb_branch(rgb), 2))
        rgb_branch = F.relu(F.max_pool2d(self.conv2_rgb_branch(rgb_branch), 2))
        rgb_branch = F.relu(F.max_pool2d(self.conv3_rgb_branch(rgb_branch), 2))
        rgb_branch = F.relu(F.max_pool2d(self.conv4_rgb_branch(rgb_branch), 2))
        rgb_branch = torch.flatten(rgb_branch)

        occ_branch = F.relu(F.max_pool2d(self.conv1_occ_branch(occ_plus), 2))
        occ_branch = F.relu(F.max_pool2d(self.conv2_occ_branch(occ_branch), 2))
        occ_branch = F.relu(F.max_pool2d(self.conv3_occ_branch(occ_branch), 2))
        occ_branch = F.relu(F.max_pool2d(self.conv4_occ_branch(occ_branch), 2))
        occ_branch = torch.flatten(occ_branch)

        concatenated = torch.cat([rgb_branch, occ_branch], dim=0)

        output = F.relu(self.fc1(concatenated))
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))

        return output




