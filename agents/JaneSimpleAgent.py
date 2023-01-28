import scipy.misc
import pygame
from agents.navigation.local_planner import RoadOption
import carla

from srunner.challenge.autoagents.autonomous_agent import AutonomousAgent
import srunner.challenge.autoagents.JaneSimpleModel as Jane                   # The Jane Pytorch Model.
import srunner.challenge.autoagents.JaneHazardModel as HazardModel                   # The Jane Pytorch Model.

import cv2
import time
import math
import os                                 # system commands
import sys                                # system commands
import json                               # read measurement.json files
import torch                              # pytorch machine learning lib
import random                             # random lib
import numpy as np                        # numerical python lib
import torch.nn as nn                     # pytorch nueral network

def distance_vehicle(waypoint, vehicle_position):

    dx = waypoint['lat'] - vehicle_position[0]
    dy = waypoint['lon'] - vehicle_position[1]

    return math.sqrt(dx * dx + dy * dy)

class JaneSimpleAgent(AutonomousAgent):
    """
    Jane Simple Agent meant to be used with the challenge_evaluator
    """
    def setup(self, path_to_conf_file):
        # visualization via pygame
        self.gps_route = []
        self.visualize = True
        self.viz_fps = 5

        self.right_ctr = 0
        GPU_NUM = 0 #NUMBER BETWEEN 0 AND 7 INCLUSIVE
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(GPU_NUM)
        self.current_device = torch.cuda.current_device()
        self.device_count   = torch.cuda.device_count() - 1 #indexing starts at zero.
        self.gpu_name       = torch.cuda.get_device_name(0)

        print("This model will run on GPU {}/{}: {}".format(self.current_device, self.device_count, self.gpu_name))

        self._image_size = (88, 200, 3)
        self._image_cut = [115, 510]
        self.flag = 1

        self.model = Jane.Jane(3)
        self.model = nn.DataParallel(self.model)
        self.model_path = '../torch_models/Jane_torch_v9_1000'
        self.model.load_state_dict(torch.load(self.model_path))
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Loaded model:", self.model_path)

        self.model_hazard = HazardModel.HazardModel()
        self.model_hazard = nn.DataParallel(self.model_hazard)
        self.model_hazard_path='../torch_models/Jane_hazard_v3_50'
        self.model_hazard.load_state_dict(torch.load(self.model_hazard_path))
        self.model_hazard = self.model_hazard.to(self.device)
        self.model_hazard.eval()

        print("Loaded model:", self.model_hazard_path)
        print(os.getcwd())


        self.direction = 'lane_follow'
        if self.visualize:
            print("Visualization Enabled.")
            self.quit = False
            self.WIDTH = 800
            self.HEIGHT = 600
            self.WIDTH = 200
            self.HEIGHT = 88
            pygame.init()
            pygame.font.init()
            self._clock = pygame.time.Clock()
            self._display = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame.display.set_caption("Jane Agent")

    def sensors(self):
        """
        Define the sensor suite required by the agent
        """
        sensors = [{'type': 'sensor.camera.rgb',
                   'x': 2.0, 'y': 0.0,
                    'z': 1.40, 'roll': 0.0,
                    'pitch': 0.0, 'yaw': 0.0,
                    'width': 800, 'height': 600,
                    'fov': 90,
                    'id': 'rgb'},
                   {'type': 'sensor.speedometer',
                    'reading_frequency': 25,
                    'id': 'speed'
                    },
                    {'type': 'sensor.other.gnss',
                     'x': 0.7, 'y': -0.4, 'z': 1.60,
                     'id': 'GPS'}
                  ]

        return sensors

    def run_step(self, input_data):
        rgb_image = np.array(input_data['rgb'][1])
        rgb_image = rgb_image[self._image_cut[0]:self._image_cut[1], :, :]  # crop

        speed = np.array(input_data['speed'][1])
        speed = speed * 3.6 / 85.0

        image_input = scipy.misc.imresize(rgb_image, [self._image_size[0],
                                                      self._image_size[1]])

        # VISUALIZE
        if self.visualize:
            if input_data['rgb'][0] % self.viz_fps == 0:
                self._surface = pygame.surfarray.make_surface(image_input.swapaxes(0,1))
                if self._surface is not None:
                    self._display.blit(self._surface, (0,0))
                pygame.display.flip()
                if self.quit:
                    pygame.quit()
        image_input = image_input.astype(np.float32)
        image_input = np.multiply(image_input, 1.0 / 255.0)
        image_input = image_input.reshape(
                (1, self._image_size[0], self._image_size[1], self._image_size[2]))



        if self.flag:
            self.car_start_lat = input_data['GPS'][1][0]
            self.car_start_lon = input_data['GPS'][1][1]
            plan = self._global_plan
            waypoint_lat = plan[0][0]['lat']
            waypoint_lon = plan[0][0]['lon']
            self.diff_lat = self.car_start_lat - waypoint_lat
            self.diff_lon = self.car_start_lon - waypoint_lon
            self.flag = False

        loc = input_data['GPS'][1]
        actual_pos = [loc[0] - self.diff_lat, loc[1] - self.diff_lon, loc[2]]
        self.gps_route.append((actual_pos[0], actual_pos[1]))
        directions = self._get_current_direction(actual_pos)
        speed = speed.astype(np.float32)
        speed = speed.reshape(1,)


        # prediction = self.model.predict([image_input])[0]
        # hazard_prediction = self.model_hazard.predict([image_input])[0]
        # prediction = np.array(prediction).reshape(2)
        # print(prediction)
        curr_img = torch.from_numpy(image_input).permute(0,3,1,2).type('torch.FloatTensor').to(self.device)
        speed = torch.from_numpy(speed).type('torch.FloatTensor').to(self.device).reshape(-1, 1)
        direc = torch.from_numpy(np.array(directions)).type("torch.FloatTensor").to(self.device).reshape(-1, 1)
        # print(type(self.curr_set))
        # output = self.model(curr_img,speed,direc)
        # print(self.model(curr_img,speed,direc))
        print(curr_img.size())
        pred_stop_traffic_light, pred_stop_vehicle, pred_stop_pedestrian = self.model_hazard(curr_img)
        stop_traffic_light = pred_stop_traffic_light.cpu().detach().numpy()[0]
        stop_vehicle = pred_stop_vehicle.cpu().detach().numpy()[0]
        stop_pedestrian = pred_stop_pedestrian.cpu().detach().numpy()[0]
        hazard_flag = stop_traffic_light > 0.5 or \
                                stop_vehicle > 0.5 or \
                                stop_pedestrian > 0.5
        if hazard_flag:
            throttle = 0.0
            brake = 1.0
            steer = 0.0
        else:
            output_steer, output_throttle, output_brake = self.model(curr_img,speed,direc)
            prediction = [0, 0, 0]
            prediction[0] = output_steer.cpu().detach().numpy()[0]
            prediction[1] = output_throttle.cpu().detach().numpy()[0]
            prediction[2] = output_brake.cpu().detach().numpy()[0]
            steer = prediction[0]
            throttle = np.clip(prediction[1], 0, .5)
            brake = prediction[2]

            ##### HEURISTICS ######
            if directions == 2.0:
                throttle = 0.5
            else:
                throttle = 0.35
            # if directions == 3.0: #left
            #     steer *= 1.5
            # elif directions == 4.0: # right
            #     steer *= 1.2
            # if np.max(image_input[:, :, :, 0]) > 0.95:
            #     brake = 1.0
            #     throttle = 0.0
            # else:
            brake = 0.0
        hand_brake = False



        # if input_data['rgb'][0] % self.viz_fps == 0:
        #     print("====================================================")
        #     # print(np.round(hazard_prediction, 2))
        #     # if hazard_prediction[3] < 0.4:
        #     #     print("Safe")
        #     # else:
        #     #     print("Red Traffic Signal or/and Vehicle Ahead")
        #     print("NN out (Steer, Throttle, Brake):", steer, throttle, brake)
        #     print("NN out (Speed, Handbrake):", speed[0], hand_brake)

        # RETURN CONTROL
        control = carla.VehicleControl()
        control.steer = float(steer)
        control.throttle = float(throttle)
        control.brake = float(brake)
        control.hand_brake = hand_brake
        # print(control)
        # print("Steer:", steer)
        # print("Throttle:, ")
        if input_data['rgb'][0] % self.viz_fps == 0:
            print("====================================================")
            print("(Frame, Direction):", input_data['rgb'][0], directions)
            print("(Steer, Throttle, Brake):", steer, throttle, brake)
            print("(Speed, Handbrake):", speed[0], hand_brake)
            print("(Stop Vehicle, Red Light, Pedestrian)", np.round(stop_traffic_light, 2),
                                                            np.round(stop_vehicle, 2),
                                                            np.round(stop_pedestrian, 2))
        return control

    def _get_current_direction(self, vehicle_position):

        # for the current position and orientation try to get the closest one from the waypoints
        closest_id = 0
        switched = 0
        min_distance = 100000
        for index in range(len(self._global_plan)):

            waypoint = self._global_plan[index][0]
            # dir      = self._global_plan[index][1]
            # if dir == RoadOption.LEFT or dir == RoadOption.RIGHT:
            computed_distance = distance_vehicle(waypoint, vehicle_position)
                # print("computed_distance:", computed_distance, dir)
            if computed_distance < min_distance:
                min_distance = computed_distance
                closest_id = index

            # waypoint = self._global_plan[index][0]
            # computed_distance = distance_vehicle(waypoint, vehicle_position)
            # if computed_distance < min_distance:
            #     min_distance = computed_distance
            #     closest_id = index

        # print ("Closest waypoint ", closest_id, "dist ", min_distance)
        direction = self._global_plan[closest_id][1]
        # if not switched:
        #     direction = 'lane_follow'

        if direction == RoadOption.LEFT:
            direction = 3.
        elif direction == RoadOption.RIGHT:
            direction = 4.
        elif direction == RoadOption.STRAIGHT:
            direction = 5.
        elif direction == RoadOption.LANEFOLLOW:
            direction = 2.
        elif direction == RoadOption.CHANGELANELEFT:
            direction = 2.
        elif direction == RoadOption.CHANGELANERIGHT:
            direction = 2.
        else:
            direction = 2.
        # print(direction)
        return direction

    def save(self, name):
        gps_np = np.array(self.gps_route)
        np.savetxt(name + '.csv', gps_np, delimiter=',' )
