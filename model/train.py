import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import numpy as np
import wandb
from cnn_rnn import Net, PrimNet
from dataloaders_prim import CIS700Dataset, CIS700Pickled
from torch.utils.data import DataLoader
import time
import sys
import cv2
import pickle
import random
from datetime import datetime
import os

# eventually we can do sweeps with this setup
hyperparameter_defaults = dict(
    batch_size=48,
    learning_rate=0.001,
    weight_decay=0.0005,
    epochs=20,
    test_iters=50,
    num_workers=48,
    # num_workers=16,
    map_size=20,
    loaders_from_scratch=True,
    test_only=False,
    weight_val=10,
    samples_per_second=4,
    pickle_batches=False,
    primitives=True,
    rollout=3,
    n_primitives=25
)

dt = datetime.now().strftime("%m_%d_%H_%M")
name_str = "_prim_weighted_rollout_bce"
wandb.init(project="cis700", config=hyperparameter_defaults, name=dt + name_str)
config = wandb.config

backup_dir = "models_v6/" + dt + name_str

os.makedirs(backup_dir, exist_ok=True)

net = Net().cuda().float()
if config.primitives:
    net = PrimNet().cuda().float()

# the usual suspects
optimizer = optim.Adam(net.parameters(), lr=config.learning_rate, betas=(0.9, 0.999), eps=1e-8,
                       weight_decay=config.weight_decay, amsgrad=False)


# criterion = nn.L1Loss(reduction="sum")


def weighted_mse(output, target):
    weight = np.ones(target.shape)
    weight[target == 1] = 100

    weight = torch.from_numpy(weight)
    loss = torch.sum(weight * (output - target) ** 2)
    return loss


def weighted_l1(output, target, weight_val=config.weight_val):
    weight = np.ones(target.shape)
    weight[target != 0] = weight_val
    weight = torch.from_numpy(weight)

    loss = torch.sum(weight * torch.abs(output - target))
    return loss


def weighted_bce(output, target, weight_val=config.weight_val):
    weight = np.ones(target.shape)
    weight[target != 0] = weight_val
    weight = torch.from_numpy(weight)
    eps = 0.0001
    # print(torch.log(output + eps))
    # print(torch.log(1 - output + eps))
    # print(target.shape)
    # print(torch.max(target), torch.min(target))
    # print(output.shape)
    bce = -(target * weight * torch.log(output + eps) + (1 - target) * torch.log(1 - output + eps))

    return torch.sum(bce)


def weighted_combo(output, target, weight_val=config.weight_val):
    return weighted_bce(output, target, weight_val) + weighted_l1(output, target, weight_val)


def best_of_many_weighted_bce(output, target, valid, weight_val=config.weight_val):
    target = target[:, None, :]
    weight = np.ones(target.shape)
    weight[target != 0] = weight_val
    weight = torch.from_numpy(weight)
    eps = 0.0001
    # print(torch.log(output + eps))
    # print(torch.log(1 - output + eps))
    # print(torch.max(target), torch.min(target))
    bce = -(target * weight * torch.log(output + eps) + (1 - target) * torch.log(1 - output + eps))[valid, :]
    print(bce.shape)
    sums = torch.sum(bce, [0, 2, 3])

    return torch.min(sums)


cel = nn.CrossEntropyLoss()
def prim_bce(output, target, rollout=config.rollout, weight_val=config.weight_val):
    loss = 0
    for i in range(rollout):
        loss += weight_val * (i / rollout) * cel(output[:, :, i].float(), target[:, i].long())

    return loss

# criterion = weighted_l1
# criterion = weighted_bce
if config.primitives:
    criterion = prim_bce
else:
    criterion = best_of_many_weighted_bce
# criterion = weighted_combo

test_filename_stub = "imgs/epoch_{}_{}.png"
train_filename_stub = "imgs/epoch_{}_{}_{}.png"

train_loaders = []
test_loaders = []

if config.loaders_from_scratch:
    # instantiate the datasets
    train_sub_dirs = [
        # "/media/ian/SSD1/tmp_datasets/UnityLearnedPlanning/20201204-015130/",
        # "/media/ian/SSD1/tmp_datasets/UnityLearnedPlanning/20201204-015728/",
        # "/media/ian/SSD1/tmp_datasets/UnityLearnedPlanning/20201204-015809/",
        # "/media/ian/SSD1/tmp_datasets/UnityLearnedPlanning/20201204-015930/",
        # "/media/ian/SSD1/tmp_datasets/UnityLearnedPlanning/20201204-020050/",
        # "/media/ian/SSD1/tmp_datasets/UnityLearnedPlanning/20201204-020212/",
        # "/media/ian/SSD1/tmp_datasets/UnityLearnedPlanning/20201204-020334/",
        # "/media/ian/SSD1/tmp_datasets/UnityLearnedPlanning/20201204-020456/",
        # "/media/ian/SSD1/tmp_datasets/UnityLearnedPlanning/20201204-020535/",
        # "/media/ian/SSD1/tmp_datasets/UnityLearnedPlanning/20201204-020635/",
        # "/media/ian/SSD1/tmp_datasets/UnityLearnedPlanning/20201204-020722/",
        # "/media/ian/SSD1/tmp_datasets/UnityLearnedPlanning/20201205-130750/",
        # "/media/ian/SSD1/tmp_datasets/UnityLearnedPlanning/20201205-130953/",
        # "/media/ian/SSD1/tmp_datasets/UnityLearnedPlanning/20201205-131659/",
        # "/media/ian/SSD1/tmp_datasets/UnityLearnedPlanning/20201205-131845/",
        # "/media/ian/SSD1/tmp_datasets/UnityLearnedPlanning/20201205-131958/"

        # "/media/ian/SSD1/tmp_datasets/UnityLearnedPlanning/20201123-191213/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201123-191213/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201124-003314/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201123-234438/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201124-034755/",

        "/home/adarsh/HDD1/cis700_final/processed/20201204-015130/",
        "/home/adarsh/HDD1/cis700_final/processed/20201204-015728/",
        "/home/adarsh/HDD1/cis700_final/processed/20201204-015809/",
        "/home/adarsh/HDD1/cis700_final/processed/20201204-015930/",
        "/home/adarsh/HDD1/cis700_final/processed/20201204-020050/",
        "/home/adarsh/HDD1/cis700_final/processed/20201204-020212/",
        "/home/adarsh/HDD1/cis700_final/processed/20201204-020334/",
        "/home/adarsh/HDD1/cis700_final/processed/20201204-020456/",
        "/home/adarsh/HDD1/cis700_final/processed/20201204-020535/",
        "/home/adarsh/HDD1/cis700_final/processed/20201204-020635/",

        # "/home/adarsh/HDD1/cis700_final/processed/20201124-034255/",
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
    # 20201124-003314.bag, 20201123-234438.bag, 20201124-034755.bag

    test_sub_dirs = [
        # "/media/ian/SSD1/tmp_datasets/UnityLearnedPlanning/20201204-020722/"
        # "/media/ian/SSD1/tmp_datasets/UnityLearnedPlanning/20201124-034255/",
        # "/media/ian/SSD1/tmp_datasets/UnityLearnedPlanning/20201124-034255/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201124-034255/",
        # "/home/adarsh/HDD1/cis700_final/processed/20201204-015130/",
    ]

    # config_file = "/home/ian/catkin/cis700_ws/src/rosbag-dl-utils/harvester_configs/cis700_prim.yaml"

    # config_file = "/home/ian/catkin/cis700_ws/src/rosbag-dl-utils/harvester_configs/cis700.yaml"
    config_file = "/home/adarsh/ros-workspaces/cis700_workspace/src/rosbag-dl-utils/harvester_configs/cis700.yaml"
    if not config.test_only:
        for sdir in train_sub_dirs:
            print(sdir)
            # no idea why it fails sometimes, here's a cheap hack
            try:
                train_loaders.append(DataLoader(
                    CIS700Dataset(config_file, sdir, samples_per_second=config.samples_per_second,
                                  map_size=config.map_size, rollout=config.rollout),
                    batch_size=config.batch_size,
                    num_workers=config.num_workers, shuffle=True))
            except ValueError:
                train_loaders.append(DataLoader(
                    CIS700Dataset(config_file, sdir, samples_per_second=config.samples_per_second,
                                  map_size=config.map_size, rollout=config.rollout),
                    batch_size=config.batch_size,
                    num_workers=config.num_workers, shuffle=True))
            # pickle.dump(train_loaders, open("train_loaders.pkl", 'wb'))

    for sdir in test_sub_dirs:
        # no idea why it fails sometimes, here's a cheap hack
        print(sdir)
        try:
            test_loaders.append(DataLoader(
                CIS700Dataset(config_file, sdir, samples_per_second=config.samples_per_second,
                              map_size=config.map_size, rollout=config.rollout),
                batch_size=1, shuffle=False))
        except ValueError:
            test_loaders.append(DataLoader(
                CIS700Dataset(config_file, sdir, samples_per_second=config.samples_per_second,
                              map_size=config.map_size, rollout=config.rollout),
                batch_size=1, shuffle=False))

        # pickle.dump(test_loaders, open("test_loaders.pkl", 'wb'))
else:
    if not config.test_only:
        for sdir in train_sub_dirs:
            train_loaders.append(DataLoader(CIS700Pickled(sdir.replace("processed", "pickled")),
                                            batch_size=1,
                                            num_workers=config.num_workers, shuffle=True))

    for sdir in test_sub_dirs:
        test_loaders.append(DataLoader(
            CIS700Dataset(config_file, sdir, samples_per_second=config.samples_per_second,
                          map_size=config.map_size),
            batch_size=1, shuffle=True))


# if config.test_only:
#     net.load_state_dict(
#         torch.load(
#             "/home/adarsh/ros-workspaces/cis700_workspace/src/cis700/model/models/model_0.ckpt"))

# let's do and save some viz stuff
def torch_to_cv2(out, single_channel=False):
    # print(out.shape)
    # print(torch.min(out), torch.max(out))
    rand_batch_idx = int(np.random.random() * out.shape[0])
    out = out.numpy()[rand_batch_idx, :, :, :]
    out -= np.min(out)
    out *= 255.0 / (np.max(out))
    out = np.moveaxis(out, 0, 2)

    if single_channel:
        return out[:, :, 1]

    if out.shape[2] < 3:
        out = np.append(out, np.zeros((out.shape[0], out.shape[1], 1)), axis=2)
    return out


# train !
for epoch in range(config.epochs):
    if not config.test_only:
        random.shuffle(train_loaders)
        for train_loader in train_loaders:
            for i_batch, sample_batched in enumerate(train_loader):

                # if config.pickle_batches:
                #     pickle_dir = train_loader.dataset.data_dir.replace("processed", "pickled")
                #     os.makedirs(pickle_dir, exist_ok=True)
                #     pickle_fname = (pickle_dir + "{:0>6d}.pkl").format(i_batch)
                #     pickle.dump(sample_batched, open(pickle_fname, 'wb'))
                #     print("dumped", pickle_fname)
                #     continue

                t1 = time.time()
                annotated, rgb, semantic, out, mps, valid = sample_batched

                optimizer.zero_grad()  # zero the gradient buffers

                # numpy to float tensor and all that junk
                annotated_tensor = annotated.float() + 1
                rgb_tensor = rgb.float() + 1
                semantic_tensor = semantic.float()
                out_tensor = out.float()
                mps_tensor = mps.float()

                # forward! doesn't use semantic rn but we have it i guess
                output = net(rgb_tensor.cuda(), annotated_tensor.cuda())
                # print(output.shape)

                # print(torch.min(out[:,1,:,:]))
                # print(torch.max(out[:,1,:,:]))
                # print('================')
                # print(torch.min(annotated[:,1,:,:]))
                # print(torch.max(annotated[:,1,:,:]))

                if config.primitives:
                    loss = criterion(output.cpu().float(), mps_tensor)
                else:
                    loss = criterion(output.cpu().float()[:, :, 1, :, :], out_tensor[:, 1, :, :], valid)
                # loss = criterion(output.cpu().float()[:, :, 1, :, :], annotated_tensor[:, 1, :, :], valid)

                wandb.log({'epoch': epoch, 'iteration': i_batch, 'loss': loss.item(),
                           'loader': train_loader.dataset.data_dir})
                print({'epoch': epoch, 'iteration': i_batch, 'loss': loss.item(),
                       'loader': train_loader.dataset.data_dir})

                # backprop
                loss.backward()
                optimizer.step()  # Does the update

                backup_path = backup_dir + "/model.ckpt"

                torch.save(net.state_dict(), backup_path)
                t2 = time.time()

                if i_batch % 50 == 0 and (torch.any(valid) or config.primitives):
                    rgb_gt = torch_to_cv2(rgb[valid, :])
                    cv2.imwrite(train_filename_stub.format(epoch, 0, "train_rgb"), rgb_gt)
                    wandb.log({"train_rgb": [wandb.Image(rgb_gt, caption=str(epoch))]})

                    if not config.primitives:
                        out_gt = torch_to_cv2(out[valid, :])
                        cv2.imwrite(train_filename_stub.format(epoch, 0, "train_out_gt"), out_gt)
                        wandb.log({"train_out_gt": [wandb.Image(out_gt, caption=str(epoch))]})

                    annotated_disp = torch_to_cv2(annotated[valid, :])
                    cv2.imwrite(train_filename_stub.format(epoch, 0, "train_annotated"), annotated_disp)
                    wandb.log({"train_annotated": [wandb.Image(annotated_disp, caption=str(epoch))]})

                    for i in range(output.shape[1]):
                        name = "train_out_pred" + str(i)
                        if not config.primitives:
                            out_pred = torch_to_cv2(output[valid, i, :].cpu().detach().float(), single_channel=True)
                            cv2.imwrite(train_filename_stub.format(epoch, 0, name), out_pred)
                            wandb.log({name: [wandb.Image(out_pred, caption=str(epoch))]})

                    cv2.imshow("train_rgb", rgb_gt / 255)
                    if not config.primitives:
                        cv2.imshow("out_pred", out_pred / 255)
                        cv2.imshow("out_gt", out_gt)
                    else:
                        print("Pred: ", output[0])
                        print("Gt: ", mps_tensor[0])
                    cv2.imshow("annotated", annotated_disp)

                    cv2.waitKey(100)

    random.shuffle(test_loaders)
    for test_loader in test_loaders:
        losses = 0
        for i_batch, sample_batched in enumerate(test_loader):
            if i_batch > config.test_iters:
                break

            annotated, rgb, semantic, out, mps, valid = sample_batched

            # numpy to float tensor and all that junk
            annotated_tensor = annotated.float() + 1
            rgb_tensor = rgb.float() + 1
            semantic_tensor = semantic.float()
            out_tensor = out.float()
            mps_tensor = mps.float()

            # forward! doesn't use semantic rn but we have it i guess
            output = net(rgb_tensor.cuda(), annotated_tensor.cuda())

            # hack because i like big numbers
            loss = criterion(output.cpu().float()[:, :, 1, :, :], out_tensor[:, 1, :, :], valid)
            losses += loss.item()

            rgb_gt = torch_to_cv2(rgb)
            cv2.imwrite(test_filename_stub.format(epoch, "test_rgb"), rgb_gt)
            wandb.log({"test_rgb": [wandb.Image(rgb_gt, caption=str(epoch))]})

            out_gt = torch_to_cv2(out)
            cv2.imwrite(test_filename_stub.format(epoch, "test_out_gt"), out_gt)
            wandb.log({"test_out_gt": [wandb.Image(out_gt, caption=str(epoch))]})

            annotated_disp = torch_to_cv2(annotated)
            cv2.imwrite(test_filename_stub.format(epoch, "test_annotated"), annotated_disp)
            wandb.log({"test_annotated": [wandb.Image(annotated, caption=str(epoch))]})

            for i in range(output.shape[1]):
                name = "test_out_pred" + str(i)
                out_pred = torch_to_cv2(output[:, i, :].cpu().detach().float(), single_channel=True)
                cv2.imwrite(train_filename_stub.format(epoch, 0, name), out_pred)
                wandb.log({name: [wandb.Image(out_pred, caption=str(epoch))]})

            wandb.log({'test_loss': losses / config.test_iters})
            print({'test_loss': loss})

            # if i_batch == 0:
            # cv2.imshow("out_gt", out_gt)
            # cv2.imshow("annotated", annotated_disp)
            # cv2.imshow("out_pred", out_pred)
            # cv2.waitKey(1000)

    PATH = "models/model_{}.ckpt".format(epoch)
    torch.save(net.state_dict(), PATH)
    wandb.save(PATH)
