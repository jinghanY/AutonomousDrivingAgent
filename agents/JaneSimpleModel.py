import os                                 # system commands
import sys                                # system commands
import json                               # read measurement.json files
import torch                              # pytorch machine learning lib
import random                             # random lib
import numpy as np                        # numerical python lib
import torch.nn as nn                     # pytorch nueral network
from PIL import Image                     # image processing lib
import matplotlib.pyplot as plt           # plotting lib
from timeit import default_timer as timer # time lib

class Jane(nn.Module):
    def __init__(self, output_dim):
        super(Jane, self).__init__()


        # input_dim = (3,88,200)
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=4),
            nn.BatchNorm2d(num_features=32),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True)
#             nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True)
#             nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # input_dim = (24,44,100)

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True))
#             nn.MaxPool2d(kernel_size=2, stride=2))

        #input_dim = (36,22,50)

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True))

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True))

        self.layer6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True))
        #input_dim = (48, 11, 25)


        self.layer7 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(num_features=256),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True)
        )

        #input_dim = (64, 11, 25)
        self.layer8 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=2),
            nn.BatchNorm2d(num_features=256),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True)
        )
        #out_dim = (64,11,25)

        self.fc1_speed = nn.Sequential(
                 nn.Linear(1, 128),
                 nn.Dropout(p=0.5),
                 nn.ReLU()
        )

        self.fc2_speed = nn.Sequential(
                 nn.Linear(128, 128),
                 nn.Dropout(p=0.5),
                 nn.ReLU()
        )

        self.fc1 = nn.Sequential(
#             nn.Linear(67456, 512),
            nn.Linear(188160, 512),
            nn.Dropout(p=0.3),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features = 512, out_features=512),
            nn.Dropout(p=0.3),
            nn.ReLU()
        )

        self.fc3 = nn.Sequential(
            nn.Linear(in_features = 640, out_features=256),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )

        self.lanefollow = nn.Linear(in_features=256, out_features=256)
        self.right = nn.Linear(in_features=256, out_features=256)
        self.left = nn.Linear(in_features=256, out_features=256)
        self.straight = nn.Linear(in_features=256, out_features=256)

        self.fc4_lane = nn.Sequential(
            nn.Linear(in_features = 256, out_features=256),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )
        self.fc5_lane = nn.Sequential(
            nn.Linear(in_features = 256, out_features=output_dim)
        )

        self.fc4_right = nn.Sequential(
            nn.Linear(in_features = 256, out_features=256),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )
        self.fc5_right = nn.Sequential(
            nn.Linear(in_features = 256, out_features=output_dim)
        )

        self.fc4_left = nn.Sequential(
            nn.Linear(in_features = 256, out_features=256),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )
        self.fc5_left = nn.Sequential(
            nn.Linear(in_features = 256, out_features=output_dim)
        )

        self.fc4_straight = nn.Sequential(
            nn.Linear(in_features = 256, out_features=256),
            nn.Dropout(p=0.5),
            nn.ReLU()
        )
        self.fc5_straight = nn.Sequential(
            nn.Linear(in_features = 256, out_features=output_dim)
        )


    def forward(self, x=torch.zeros(1,88,200,3), speed=0, command=3):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc1(out)
        out = self.fc2(out)

        out_speed = self.fc1_speed(speed)
        out_speed = self.fc2_speed(out_speed)
        out = torch.cat([out, out_speed], 1)

        out = self.fc3(out)

        left     = self.left(out)
        left     = self.fc4_left(left)
        left     = self.fc5_left(left)

        right     = self.right(out)
        right     = self.fc4_right(right)
        right     = self.fc5_right(right)

        lane     = self.lanefollow(out)
        lane     = self.fc4_lane(lane)
        lane     = self.fc5_lane(lane)

        straight     = self.straight(out)
        straight     = self.fc4_straight(straight)
        straight     = self.fc5_straight(straight)

        out = torch.where(torch.eq(command, 2.0), lane,
                   (torch.where(torch.eq(command, 3.0), right,
                   (torch.where(torch.eq(command, 4.0), left, straight)))))

        steer, throttle, brake = out[:, 0], out[:, 1], out[:, 2]

        return steer, throttle, brake
