import sys
sys.path.append("./drive/yolo2")
sys.path.append("../scenario_runner")
sys.path.append("../../scenario_runner")
import torch
import carla
import pygame
import scipy.misc
import numpy as np
import torch.nn as nn
from drive.yolo2.utils import *
import matplotlib.pyplot as plt
import drive.JaneSimpleModel as Jane
from drive.yolo2.darknet import Darknet
from agents.navigation.local_planner import RoadOption




def distance_vehicle(waypoint, vehicle_position):

    dx = waypoint['lat'] - vehicle_position[0]
    dy = waypoint['lon'] - vehicle_position[1]
    return np.sqrt(dx * dx + dy * dy)

class JaneSimpleAgent(object):


    def __init__(self):

        ### CONSTANTS #################################################################################################################
        ## speed section ##
        self.convert_mps_to_kmh         = 3.6  # (3600 seconds / hr) * (1 km / 1000 m).
        self.approx_max_carla_speed_kmh = 85.0 # approximate maximum speed in the carla simulator in kmh.

        ## image section ##
        self.max_color_value = 255.0
        self._image_size     = (88, 200, 3)
        self._image_cut      = [115, 510]

        ## yolo section ##
        self.confidence_threshold          = 0.5  # confidence for yolo model
        self.nms_threshold                 = 0.4  # nms threshold for yolo model
        self.confidence_heuristic          = 0.95 # our heuristic confidence threshold
        self.objects_of_interest           = ['car', 'person', 'truck', 'motorbike', 'bicycle'] # subset of YOLOv2 objects
        self.max_r_channel_value           = 0.95
        self.max_x_loc                     = 0.6
        self.min_x_loc                     = 0.4
        self.max_brake_frame_traffic_light = 10
        self.max_brake_frame_vehicle       = 15
        self.max_brake_frame_pedestrian    = 15
        self.brake_frames                  = 0

        ## visualize section ##
        self.visualize = True
        self.viz_fps = 5
        self.curr_frame = 0

        ## driving model section ##
        self.throttle_lane_follow     = 0.52
        self.throttle_intersection    = 0.40
        self.steer_left_scale_factor  = 1.7
        self.steer_right_scale_factor = 1.2
        self.max_Jane_speed           = 0.85
        self.exp_moving_alpha         = 0.9
        self.prev_steer               = 0.0
        self.flag                     = 1
        self.num_outputs              = 3

        ## model paths section ##
        self.model_path        = '../torch_models/Jane_torch_v10_475'
        self.modelTurn_path    = '../torch_models/Jane_torch_v13_steer_2000'
        self.hazard_cfgfile    = './drive/yolo2/cfg/yolo.cfg'
        self.hazard_wgtfile    = './drive/yolo2/yolo.weights'
        self.hazard_names_voc  = './drive/yolo2/data/voc.names'
        self.hazard_names_coco = './drive/yolo2/data/coco.names'
        self.hazard_names      = './drive/yolo2/data/names'
        ###############################################################################################################################


        GPU_NUM = 0 #NUMBER BETWEEN 0 AND 7 INCLUSIVE
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(GPU_NUM)
        self.current_device = torch.cuda.current_device()
        self.device_count   = torch.cuda.device_count() - 1 #indexing starts at zero.
        self.gpu_name       = torch.cuda.get_device_name(0)
        print("This model will run on GPU {}/{}: {}".format(self.current_device, self.device_count, self.gpu_name))


        # lane_follow model
        self.model = Jane.Jane(self.num_outputs)
        self.model = nn.DataParallel(self.model)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Loaded model:", self.model_path)

        # intersection model
        self.modelTurn = Jane.Jane(self.num_outputs)
        self.modelTurn = nn.DataParallel(self.modelTurn)
        self.modelTurn.load_state_dict(torch.load(self.modelTurn_path))
        self.modelTurn = self.modelTurn.to(self.device)
        self.modelTurn.eval()
        print("Loaded model:", self.modelTurn_path)

        # hazard model
        self.hazard_yolo = Darknet(self.hazard_cfgfile)
        self.hazard_yolo.load_weights(self.hazard_wgtfile)
        if self.hazard_yolo.num_classes == 20:
            self.namesfile = self.hazard_names_voc
        elif self.hazard_yolo.num_classes == 80:
            self.namesfile = self.hazard_names_coco
        else:
            self.namesfile = self.hazard_names
        self.hazard_yolo.cuda()
        self.class_names = load_class_names(self.namesfile)
        print("Loaded model:", self.hazard_wgtfile)

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

    def process_sensors(self, image, speed, direction):
        """
        process raw sensor data from CARLA into Jane model format.

        inputs:
        image     -> raw sensor.camera.rgb image from CARLA with WidthxHeight=800x600 in 8 bit int (0 to 255 range)
        speed     -> raw carla speed in meters per second
        direction -> High level command given by CARLA

        1. cut image.
        2. resize to nueral network input shape
        3. convert speed to kmh, then normalize
        4. convert all three to torch objects

        outputs:
        tuple of model inputs.
        """

        # cut image and resize to input shape of the Jane nueral network
        image = image[self._image_cut[0]:self._image_cut[1], :, :]
        image = scipy.misc.imresize(image, [self._image_size[0], self._image_size[1]])
        image = image.astype(np.float32) / self.max_color_value
        image = image.reshape((1,
                               self._image_size[0],
                               self._image_size[1],
                               self._image_size[2]))

        speed = speed * self.convert_mps_to_kmh / self.approx_max_carla_speed_kmh # normalize the speed
        speed = speed.astype(np.float32)
        speed = speed.reshape(1,)


        # convert to torch objects and send to GPU.
        final_img   = torch.from_numpy(image).permute(0,3,1,2).type('torch.FloatTensor').to(self.device)
        final_speed = torch.from_numpy(speed).type('torch.FloatTensor').to(self.device).reshape(-1, 1)
        final_direc = torch.from_numpy(np.array(direction)).type("torch.FloatTensor").to(self.device).reshape(-1, 1)

        return (final_img, final_speed, final_direc)

    def run_hazard_model(self, image, direction):
        """
        Run the hazard model and predict if the environment is safe/unsafe

        inputs:
        image     -> raw sensor.camera.rgb image from CARLA with WidthxHeight=800x600 in 8 bit int (0 to 255 range)
        direction -> High level command given by CARLA

        1. resize to YOLO nueral network input shape
        2. do detection
        3. if self.visualize, plot on pygame window
        4. predict safe/unsafe

        outputs:
        Boolean: Is the environment unsafe (True) or safe (False)?
        """

        # if we have already seen a frame that is unsafe, decrement brake_frame counter.
        if self.brake_frames:
            self.brake_frames -= 1
            return True

        # resize and detect
        image = scipy.misc.imresize(image, [self.hazard_yolo.width,self.hazard_yolo.height])
        boxes = do_detect(model=self.hazard_yolo, img=image, conf_thresh=self.confidence_threshold, nms_thresh=self.nms_threshold, use_cuda=True)
        image = plot_boxes_cv2(image, boxes, savename=None, class_names=self.class_names, color=None)

        # visualize the image
        if self.visualize:
            if self.curr_frame % self.viz_fps == 0:
                self._surface = pygame.surfarray.make_surface(scipy.misc.imresize(image, [600, 800]).swapaxes(0,1))
                if self._surface is not None:
                    self._display.blit(self._surface, (0,0))
                pygame.display.flip()
                if self.quit:
                    pygame.quit()

        # if any predictions
        if boxes:
            for box in boxes:
                if box[-2] > self.confidence_heuristic:
                    # if traffic light and r channel is high
                    if self.class_names[box[-1]] == 'traffic light' and image[:,:,0].max() > self.max_r_channel_value * self.max_color_value:
                        print(image[:,:,0].max())
                        self.brake_frames = self.max_brake_frame_traffic_light
                        return True
                    # if we see an object in our direct point of view
                    if self.class_names[box[-1]] in self.objects_of_interest:
                        if box[0] < self.max_x_loc and box[0] > self.min_x_loc and direction == 2.0:
                            self.brake_frames = self.max_brake_frame_vehicle
                            return True

        return False

    def run_driving_model(self, image, speed, direction):
        """
        Run the driving model to predict the steer. Currently, using heuristics for throttle.

        inputs:
        image     -> cut, resized, and normalized image
        speed     -> normalized speed
        direction -> High level command given by CARLA

        1. use either highway model or town model.
        2. apply heuristics to steer and throttle
        3. convert speed to kmh, then normalize
        4. convert all three to torch objects

        outputs:
        tuple of model outputs.
        """
        self.brake_frames = 0

        # lanefollow agent
        if direction == 2:
            output_steer, _, _ = self.model(image, speed, direction)
            throttle           = self.throttle_lane_follow
        # intersection agent
        else:
            output_steer, _, _ = self.modelTurn(image, speed, direction)
            throttle           = self.throttle_intersection

        # steer heuristic
        steer                = output_steer.cpu().detach().numpy()[0]

        # multiplicative steer factor
        steer_scaling_factor = (direction == 3) * self.steer_left_scale_factor \
                             + (direction == 4) * self.steer_right_scale_factor \
                             + (direction == 1) \
                             + (direction == 2)
        steer                = steer * steer_scaling_factor

        # exponential moving average for steer.
        steer                = float(self.exp_moving_alpha * steer + (1 - self.exp_moving_alpha) * self.prev_steer)
        self.prev_steer      = steer


        brake         = 0.0
        if speed > self.max_Jane_speed:
            throttle = 0.0


        prediction    = [0,0,0]
        prediction[0] = steer
        prediction[1] = throttle
        prediction[2] = brake
        return (steer, throttle, brake)

    def run_step(self, input_data):

        self.curr_frame += 1
        image           = np.array(input_data['rgb'][1])
        speed           = np.array(input_data['speed'][1])
        direction       = input_data['control']
        control         = carla.VehicleControl()


        if self.run_hazard_model(image, direction):
            brake      = 1.0
            steer      = 0.0
            throttle   = 0.0
            hand_brake = True


            control.steer      = steer
            self.prev_steer    = control.steer
            control.throttle   = float(throttle)
            control.brake      = float(brake)
            control.hand_brake = hand_brake
        else:
            final_img, final_speed, final_direc = self.process_sensors(image, speed, direction)
            steer, throttle, brake = self.run_driving_model(final_img, final_speed, final_direc)


            hand_brake         = False
            control.steer      = steer
            self.prev_steer    = control.steer
            control.throttle   = float(throttle)
            control.brake      = float(brake)
            control.hand_brake = hand_brake

        return control

    def _get_current_direction(self, vehicle_position):

        # for the current position and orientation try to get the closest one from the waypoints
        closest_id = 0
        min_distance = 100000
        for index in range(len(self._global_plan)):
            waypoint = self._global_plan[index][0]
            computed_distance = distance_vehicle(waypoint, vehicle_position)
            if computed_distance < min_distance:
                min_distance = computed_distance
                closest_id = index

        direction = self._global_plan[closest_id][1]

        if direction == RoadOption.LEFT:
            direction = 'left'
        elif direction == RoadOption.RIGHT:
            direction = 'right'
        elif direction == RoadOption.STRAIGHT:
            direction = 'straight'
        else:
            direction = 'lane_follow'
        return direction

    def save(self, name):
        gps_np = np.array(self.gps_route)
        np.savetxt(name + '.csv', gps_np, delimiter=',' )
