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
            self.conv1_occ_branch = nn.Conv2d(2, 10, kernel_size=3)
            self.conv2_occ_branch = nn.Conv2d(10, 20, kernel_size=3)
            self.conv3_occ_branch = nn.Conv2d(20, 40, kernel_size=3)
            self.conv4_occ_branch = nn.Conv2d(40, 80, kernel_size=3)

        self.fcn_1 = nn.Linear(80*2*4*4, 500)
        self.fcn_mid_1 = nn.Linear(500,500)
        self.fcn_mid_2 = nn.Linear(500,500)
        self.fcn_2 = nn.Linear(500, 80*2*4*4)

        self.t_conv1 = nn.ModuleList([])
        self.t_conv2 = nn.ModuleList([])
        self.t_conv3 = nn.ModuleList([])
        self.t_conv4 = nn.ModuleList([])
        self.t_conv5 = nn.ModuleList([])        

        self.num_preds = 1
        for _ in range(self.num_preds):
            self.t_conv1.append(nn.ConvTranspose2d(160, 80, 2, stride=2))
            self.t_conv2.append(nn.ConvTranspose2d(80, 40, 2, stride=2))
            self.t_conv3.append(nn.ConvTranspose2d(40, 20, 2, stride=2))
            self.t_conv4.append(nn.ConvTranspose2d(20*2, 10, 2, stride=2))
            self.t_conv5.append(nn.ConvTranspose2d(10, 2, 2, stride=1))

    '''
    rgb is just an rgb image, occ_plus is a 2 channel data structure with the occupancy grid, suggested path and 
    goal all encoded in the different channels
    '''
    def forward(self, rgb, occ_plus):

        if self.rgb_model_type == "other":
            # rgb encoder
            # print(rgb.shape)
            rgb_branch = F.relu(F.max_pool2d(self.conv1_rgb_branch(rgb), 2))
            # print(rgb_branch.shape)
            rgb_branch = F.relu(F.max_pool2d(self.conv2_rgb_branch(rgb_branch), 2))
            # print(rgb_branch.shape)
            rgb_branch = F.relu(F.max_pool2d(self.conv3_rgb_branch(rgb_branch), 2))
            # print(rgb_branch.shape)
            rgb_branch = F.relu(F.max_pool2d(self.conv4_rgb_branch(rgb_branch), 2))
            # print(rgb_branch.shape)
        else:
            rgb_branch = self.rgb_resnet(rgb)

        # occ encoder
        #if self.occ_model_type == "other":
        # print(occ_plus.shape)
        occ_branch = F.relu(F.max_pool2d(self.conv1_occ_branch(occ_plus), 2))
        # print(occ_branch.shape)
        occ_branch_mid = F.relu(F.max_pool2d(self.conv2_occ_branch(occ_branch), 2))
        # print(occ_branch.shape)
        occ_branch = F.relu(F.max_pool2d(self.conv3_occ_branch(occ_branch_mid), 2))
        # print(occ_branch.shape)
        occ_branch = F.relu(F.max_pool2d(self.conv4_occ_branch(occ_branch), 2))
        # print(occ_branch.shape)
        #else:
        #    occ_branch = self.occ_resnet(occ_plus)

        # print("occ_branch shape:", occ_branch.shape)
        # print("rgb_branch shape:", rgb_branch.shape)

        # put em together!
        concatenated = torch.cat([rgb_branch, occ_branch], dim=1)
        flattened = torch.flatten(concatenated, start_dim=1)
        flattened = F.relu(self.fcn_1(flattened))
        flattened = F.relu(self.fcn_mid_1(flattened))
        flattened = F.relu(self.fcn_mid_2(flattened))
        flattened = F.relu(self.fcn_2(flattened))
        concatenated = torch.reshape(flattened, concatenated.shape)
        # print("concatenated shape:", concatenated.shape)

        output_total = None
        for i in range(self.num_preds):
            output = F.relu(self.t_conv1[i](concatenated))
            # print(output.shape)
            output = F.relu(self.t_conv2[i](output))
            # print(output.shape)
            output = F.relu(self.t_conv3[i](output))

            occ_branch_mid_reshaped = F.interpolate(occ_branch_mid, (output.shape[2], output.shape[3]))
            print(output.shape)
            print(occ_branch_mid_reshaped.shape)
            output = torch.cat([output, occ_branch_mid_reshaped], dim=1)
            # print(output.shape)
            output = F.relu(self.t_conv4[i](output))

            # print(output.shape)

            # when doing regression
            # output = self.t_conv5(output)

            # when categorical
            output = F.sigmoid(self.t_conv5[i](output))
            # print(output.shape)

            # print("Output shape:", output.shape)
            output = F.interpolate(output, (occ_plus.shape[2], occ_plus.shape[3]))[:, None, :]
            # print("Interpolated shape:", output.shape)
            
            if output_total is None:
                output_total = output
            else:
                output_total = torch.cat((output_total, output), 1)

        return output_total
