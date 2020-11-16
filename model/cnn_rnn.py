import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
import torchvision
import numpy as np
import wandb


# wandb.init(project="cis700")


class Net(nn.Module):
    def __init__(self, rgb_net="other", occ_net="other"):
        super(Net, self).__init__()
        self.rgb_model_type = rgb_net
        self.occ_model_type = occ_net

        if rgb_net == 18:
            self.rgb_resnet = models.resnet18()
        elif rgb_net == 34:
            self.rgb_resnet = models.resnet34()
        elif rgb_net == 50:
            self.rgb_resnet = models.resnet50()
        elif rgb_net == 101:
            self.rgb_resnet = models.resnet101()
        elif rgb_net == 152:
            self.rgb_resnet = models.resnet152()
        elif rgb_net == "other":
            self.conv1_rgb_branch = nn.Conv2d(3, 10, kernel_size=3)
            self.conv2_rgb_branch = nn.Conv2d(10, 20, kernel_size=3)
            self.conv3_rgb_branch = nn.Conv2d(20, 40, kernel_size=3)
            self.conv4_rgb_branch = nn.Conv2d(40, 80, kernel_size=3)

        if occ_net == 18:
            self.occ_resnet = models.resnet18()
        elif occ_net == 34:
            self.occ_resnet = models.resnet34()
        elif occ_net == 50:
            self.occ_resnet = models.resnet50()
        elif occ_net == 101:
            self.occ_resnet = models.resnet101()
        elif occ_net == 152:
            self.occ_resnet = models.resnet152()
        elif occ_net == "other":
            self.conv1_occ_branch = nn.Conv2d(3, 10, kernel_size=3)
            self.conv2_occ_branch = nn.Conv2d(10, 20, kernel_size=3)
            self.conv3_occ_branch = nn.Conv2d(20, 40, kernel_size=3)
            self.conv4_occ_branch = nn.Conv2d(40, 80, kernel_size=3)

        self.t_conv1 = nn.ConvTranspose2d(160, 80, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(80, 40, 2, stride=2)
        self.t_conv3 = nn.ConvTranspose2d(40, 20, 2, stride=2)
        self.t_conv4 = nn.ConvTranspose2d(20, 10, 2, stride=2)
        self.t_conv5 = nn.ConvTranspose2d(10, 3, 2, stride=1)

    '''
    rgb is just an rgb image, occ_plus is a 3 channel data structure with the occupancy grid, suggested path and 
    goal all encoded in the different channels
    '''

    def forward(self, rgb, occ_plus):

        if self.rgb_model_type == "other":
            # rgb encoder
            rgb_branch = F.relu(F.max_pool2d(self.conv1_rgb_branch(rgb), 2))
            rgb_branch = F.relu(F.max_pool2d(self.conv2_rgb_branch(rgb_branch), 2))
            rgb_branch = F.relu(F.max_pool2d(self.conv3_rgb_branch(rgb_branch), 2))
            rgb_branch = F.relu(F.max_pool2d(self.conv4_rgb_branch(rgb_branch), 2))
        else:
            rgb_branch = self.rgb_resnet(rgb)

        # occ encoder
        if self.occ_model_type == "other":
            occ_branch = F.relu(F.max_pool2d(self.conv1_occ_branch(occ_plus), 2))
            occ_branch = F.relu(F.max_pool2d(self.conv2_occ_branch(occ_branch), 2))
            occ_branch = F.relu(F.max_pool2d(self.conv3_occ_branch(occ_branch), 2))
            occ_branch = F.relu(F.max_pool2d(self.conv4_occ_branch(occ_branch), 2))
        else:
            occ_branch = self.occ_resnet(occ_plus)

        # print("occ_branch shape:", occ_branch.shape)
        # print("rgb_branch shape:", rgb_branch.shape)

        # put em together!
        concatenated = torch.cat([rgb_branch, occ_branch], dim=1)
        # print("concatenated shape:", concatenated.shape)

        output = F.relu(self.t_conv1(concatenated))
        output = F.relu(self.t_conv2(output))
        output = F.relu(self.t_conv3(output))
        output = F.relu(self.t_conv4(output))
        output = F.relu(self.t_conv5(output))

        # print("Output shape:", output.shape)
        output = F.interpolate(output, (occ_plus.shape[2], occ_plus.shape[3]))
        # print("Interpolated shape:", output.shape)

        return output
