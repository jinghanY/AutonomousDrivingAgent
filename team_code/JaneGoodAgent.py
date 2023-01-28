import sys
TEAM_CODE_DIR = 'team_code' # change as needed
sys.path.append(TEAM_CODE_DIR)

import time                                  # system commands
import torch                              # pytorch machine learning lib
import numpy as np                        # numerical python lib
import torch.nn as nn                     # pytorch nueral network
import scipy.misc
import matplotlib.pyplot as plt

# carla related imports
from agents.navigation.local_planner import RoadOption
import carla
from srunner.challenge.autoagents.autonomous_agent import AutonomousAgent

# model imports
import JaneGoodModel as Jane                   # The Jane Pytorch Model.
import TrafficLightModel                       # Traffic Light Model
sys.path.append(TEAM_CODE_DIR + '/yolo2')       # Yolo object detection model
from darknet import Darknet
from yolo2.utils import *

# import stanley control
sys.path.append(TEAM_CODE_DIR + '/stanley')
import cubic_spline_planner
from stanley_controller import *

print("Done imports for JaneGoodAgent")

def distance_vehicle(waypoint, vehicle_position):

    dx = waypoint['lat'] - vehicle_position[0]
    dy = waypoint['lon'] - vehicle_position[1]

    return np.sqrt(dx * dx + dy * dy)

class JaneGoodAgent(AutonomousAgent):
    """
    Jane Good Agent meant to be used with the challenge_evaluator
    """
    def setup(self, path_to_conf_file):
        print("Entering JaneGoodAgent Setup Function.")
        self.brake_frames = 0
        self.visualize = False
        self.viz_fps = 1

        # setting up the GPU device
        GPU_NUM = 0 #NUMBER BETWEEN 0 AND 7 INCLUSIVE
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(GPU_NUM)
        self.current_device = torch.cuda.current_device()
        self.device_count   = torch.cuda.device_count() - 1 #indexing starts at zero.
        self.gpu_name       = torch.cuda.get_device_name(0)
        print("This model will run on GPU {}/{}: {}".format(self.current_device, self.device_count, self.gpu_name))

        # driving model image size parameters
        self._image_size = (88, 200, 3)
        self._image_cut = [115, 510]
        self.flag = 1

        # core lane follow model
        self.model = Jane.Jane(3)
        self.model = nn.DataParallel(self.model)
        self.model_path = TEAM_CODE_DIR + '/weights/Jane_torch_v9_1000'
        self.model.load_state_dict(torch.load(self.model_path))
        self.model = self.model.to(self.device)
        self.model.eval()
        print("Loaded model:", self.model_path)

        # yolo object detection model
        self.hazard_cfgfile = TEAM_CODE_DIR + '/yolo2/cfg/yolo.cfg'
        self.hazard_wgtfile = TEAM_CODE_DIR + '/yolo2/yolo.weights'
        self.hazard_yolo  = Darknet(self.hazard_cfgfile)
        self.hazard_yolo.load_weights(self.hazard_wgtfile)
        if self.hazard_yolo.num_classes == 20:
            self.namesfile = TEAM_CODE_DIR + '/yolo2/data/voc.names'
        elif self.hazard_yolo.num_classes == 80:
            self.namesfile = TEAM_CODE_DIR + '/yolo2/data/coco.names'
        else:
            self.namesfile = TEAM_CODE_DIR + '/yolo2/data/names'
        self.hazard_yolo.cuda()
        self.class_names = load_class_names(self.namesfile)
        print("Loaded model:", self.hazard_wgtfile)

        self.modelTraffic = TrafficLightModel.ZFNet()
        self.modelTraffic_path = TEAM_CODE_DIR + '/weights/weights_traffic_light_20.pth'
        self.modelTraffic.load_state_dict(torch.load(self.modelTraffic_path))
        self.modelTraffic = self.modelTraffic.to(self.device)
        self.modelTraffic.eval()
        print("Loaded model:", self.modelTraffic_path)

        self.direction = 'lane_follow'
        self.traffic_img_ctr = 0

        # setup for stanley controller
        self.do_once = True
        self.target_speed = 25 / 3.6  # [m/s]

        # visualization via pygame
        if self.visualize:
            import pygame
            print("Visualization Enabled.")
            self.quit = False
            self.WIDTH = 800
            self.HEIGHT = 600
            pygame.init()
            pygame.font.init()
            self._clock = pygame.time.Clock()
            self._display = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
            pygame.display.set_caption("JaneGoodAgent")
        print("Setup completed successfully!")

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
                    {'type': 'sensor.can_bus',
                    'reading_frequency': 25, 'id': 'can_bus'},
                    {'type': 'sensor.other.gnss',
                     'x': 0.7, 'y': -0.4, 'z': 1.60,
                     'id': 'GPS'}
                  ]
        return sensors

    def run_step(self, input_data, timestamp):
        # print("Entered run step")
        rgb_image = np.array(input_data['rgb'][1][:,:,:3][:,:,::-1]) # read image and convert BGR2RGB
        yolo_image = rgb_image.copy() # create a copy of the original for the yolo detector

        # VISUALIZE
        if self.visualize:
            if input_data['rgb'][0] % self.viz_fps == 0:
                self._surface = pygame.surfarray.make_surface(rgb_image.swapaxes(0,1))
                if self._surface is not None:
                    self._display.blit(self._surface, (0,0))
                pygame.display.flip()
                if self.quit:
                    pygame.quit()

        # DO SOMETHING SMART
        #               - CARLA TEAM

        # generate the cubic spline from the given waypoints
        if self.do_once:
            print("Doing Once.")
            ax = []
            ay = []
            ayaw = []
            for item in self._global_plan_world_coord:
                ax.append(item[0].location.x)
                ay.append(item[0].location.y)

            ax_long = []
            ay_long = []
            threshold = 2
            for i in range(1, len(ax)):
                ax_long.append(ax[i-1])
                ay_long.append(ay[i-1])
                if abs(ax[i-1] - ax[i]) < threshold or abs(ay[i-1] - ay[i]) < threshold:
                    ax_long.append((ax[i-1] + ax[i])/2)
                    ay_long.append((ay[i-1] + ay[i])/2)
            ax_long.append(ax[-1])
            ay_long.append(ay[-1])
            ax = ax_long
            ay = ay_long
            self.cx, self.cy, self.cyaw, self.ck, self.s = cubic_spline_planner.calc_spline_course(ax, ay, ds=1)
            if np.isnan(self.cx[0]): # use piecewise cublic spline calculation
                print("Failed to Calculate Route")
                print("Trying Piecewise Calculation ...")
                if len(ax) > 50:
                    split = 10
                else:
                    split = 5

                _cx = []
                _cy = []
                _cyaw = []
                _ck = []
                _s = []
                path_length = 0
                while path_length < len(ax):
                    print(path_length, split, len(ax))
                    cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax[path_length:path_length+split],
                                                                                 ay[path_length:path_length+split], ds=1)
                    _cx = _cx + cx # appending the new part of the route to the overall routes
                    _cy = _cy + cy
                    _cyaw = _cyaw + cyaw
                    _ck = _ck + ck
                    _s = _s + s
                    path_length += split
                self.cx, self.cy, self.cyaw, self.ck, self.s = _cx, _cy, _cyaw, _ck, _s
                if np.isnan(self.cx[0]):
                    print("Failed to Calculate Route")
                else:
                    print("Successfully Calculated Route")
            else:
                print("Successfully Calculated Route")
            location = input_data['can_bus'][1]['transform'].location
            rotation = input_data['can_bus'][1]['transform'].rotation
            print("Start Map (x, y, yaw)", ax[0], ay[0])
            print("Start Car (x, y, yaw)", location.x, location.y, rotation.yaw)
            self.state = State(x=location.x, y=location.y, yaw=np.radians(normalize_angle(rotation.yaw)), v=0.0)
            print("Set up state")
            self.target_idx, _ = calc_target_index(self.state, self.cx, self.cy)
            print("Done Once.")
            # plt.plot(self.cx, self.cy)
            # plt.scatter(ax, ay)
            # plt.scatter(self.state.x, self.state.y)
            # plt.show()
            self.do_once = False

        # preprocess image for the driving model
        rgb_image = rgb_image[self._image_cut[0]:self._image_cut[1], :, :3]  # crop
        image_input = scipy.misc.imresize(rgb_image, [self._image_size[0], self._image_size[1]]) # resize
        image_input = image_input.astype(np.float32) # convert to float
        image_input = np.multiply(image_input, 1.0 / 255.0) # normalize
        image_input = image_input.reshape((1, self._image_size[0], self._image_size[1], self._image_size[2])) # reshape to (1, 88, 200, 3)

        # we are fixing an error with the given GPS data (don't know if this is still needed)
        if self.flag:
            self.car_start_lat = input_data['GPS'][1][0]
            self.car_start_lon = input_data['GPS'][1][1]
            plan = self._global_plan
            waypoint_lat = plan[0][0]['lat']
            waypoint_lon = plan[0][0]['lon']
            self.diff_lat = self.car_start_lat - waypoint_lat
            self.diff_lon = self.car_start_lon - waypoint_lon
            self.flag = False
        gps_pos  = input_data['GPS'][1]
        actual_pos = [gps_pos[0] - self.diff_lat, gps_pos[1] - self.diff_lon, gps_pos[2]]

        # get direction (lanefollow, right, left, straight, changelane)
        directions = self._get_current_direction(actual_pos)

        # get speed and normalize it
        speed = np.array(input_data['can_bus'][1]['speed'])
        speed = speed * 3.6 / 85.0 # normalizing the speed to be in range [0, 1] (85 = max speed)
        speed = speed.astype(np.float32)
        speed = speed.reshape(1,)

        # prepare inputs for the hazard model
        curr_img = torch.from_numpy(image_input).permute(0,3,1,2).type('torch.FloatTensor').to(self.device)
        speed = torch.from_numpy(speed).type('torch.FloatTensor').to(self.device).reshape(-1, 1)
        direc = torch.from_numpy(np.array(directions)).type("torch.FloatTensor").to(self.device).reshape(-1, 1)

        # print(type(self.curr_set))
        # output = self.model(curr_img,speed,direc)
        # print(self.model(curr_img,speed,direc))
        if self.brake_frames > 0: # brake until brake_frames buffer is reduced to 0
            brake = 1.0
            steer = 0.0
            throttle = 0.0
            self.brake_frames -=1
            hand_brake = False
            control = carla.VehicleControl()
            control.steer = float(steer)
            control.throttle = float(throttle)
            control.brake = float(brake)
            control.hand_brake = hand_brake
            return control

        # Traffic light image preprocessing
        tl_sized = scipy.misc.imresize(yolo_image, [224, 224])
        tl_sized = tl_sized.astype(np.float32)
        tl_sized = tl_sized / 255.0
        # print(tl_sized.shape, np.max(tl_sized))
        tl_sized = tl_sized.reshape((1, 224, 224, 3))
        # print(tl_sized.shape, np.max(tl_sized))
        lights = self.modelTraffic(torch.from_numpy(tl_sized).permute(0,3,1,2).type('torch.FloatTensor').to(self.device))
        print(lights, torch.argmax(lights, dim=1).type(torch.LongTensor))

        # Traffic light Model
        if torch.argmax(lights, dim=1).type(torch.LongTensor) == 2:
            print("Red light")
            # save_text = 'red_light_{}.png'.format(input_data['rgb'][0])
            # print(save_text)
            # plt.imsave(save_text, yolo_image)
            # print("Saved image")
        else:
            print("No red light")
            # save_text = 'no_red_light_{}.png'.format(input_data['rgb'][0])
            # print(save_text)
            # plt.imsave(save_text, yolo_image)
            # print("Saved image")
        if torch.argmax(lights, dim=1).type(torch.LongTensor):
            brake = 1.0
            steer = 0.0
            throttle = 0.0
            hand_brake = True
            control = carla.VehicleControl()
            control.steer = float(steer)
            control.throttle = float(throttle)
            control.brake = float(brake)
            control.hand_brake = hand_brake
            return control

        # TODO: the following YOLO model heuristic requires restructure
        sized = scipy.misc.imresize(yolo_image, [self.hazard_yolo.width, self.hazard_yolo.height])
        boxes = do_detect(model=self.hazard_yolo, img=sized, conf_thresh=0.5, nms_thresh=0.4, use_cuda=True)
        if boxes:
            for box in boxes:
                if box[-2] > 0.95:
                    if self.class_names[box[-1]] in ['car', 'person', 'truck', 'motorbike', 'bicycle', 'traffic light']:
                        if self.class_names[box[-1]] in ['traffic light'] and image_input[:,:,:,0].max() > 0.95:
                            brake = 1.0
                            self.brake_frames = 6
                            brake = 1.0
                            steer = 0.0
                            throttle = 0.0
                            self.brake_frames -=1
                            hand_brake = False
                            control = carla.VehicleControl()
                            control.steer = float(steer)
                            control.throttle = float(throttle)
                            control.brake = float(brake)
                            control.hand_brake = hand_brake
                            return control
                        if box[0] < 0.6 and box[0] > 0.4 and directions == 2:
                            brake = 1.0
                            self.brake_frames = 20
                            brake = 1.0
                            steer = 0.0
                            throttle = 0.0
                            self.brake_frames -=1
                            hand_brake = False
                            control = carla.VehicleControl()
                            control.steer = float(steer)
                            control.throttle = float(throttle)
                            control.brake = float(brake)
                            control.hand_brake = hand_brake
                            return control

        if True: # TODO: Change condition rather than always True
            self.brake_frames = 0

            if directions == 2.0:
                print("jane: lanefollow")
                # output_steer = self.model(curr_img,speed,direc)
                output_steer, output_throttle, output_brake = self.model(curr_img,speed,direc)
                prediction = [0, 0, 0]
                prediction[0] = output_steer.cpu().detach().numpy()[0]
                # prediction[1] = output_throttle.cpu().detach().numpy()[0]
                # prediction[2] = output_brake.cpu().detach().numpy()[0]

                steer = prediction[0]
                throttle = 0.6
                brake = prediction[2]
            else:
                print("stanley: pidcontrol")
                throttle = pid_control(self.target_speed, self.state.v)
                # print("ai", ai)
                steer, self.target_idx = stanley_control(self.state, self.cx, self.cy, self.cyaw, self.target_idx)
                # print("di", di)
                self.state.x = input_data['can_bus'][1]['transform'].location.x
                self.state.y = input_data['can_bus'][1]['transform'].location.y
                self.state.yaw = normalize_angle(np.radians(input_data['can_bus'][1]['transform'].rotation.yaw))
                self.state.v = np.array(input_data['can_bus'][1]['speed'])
                steer = np.clip(steer, -0.6, 0.6)
                throttle = np.clip(throttle, 0, .7)
                brake = 0.0

            if throttle > brake:
                brake = 0.0
            if speed > 1.0 and brake == 0.0:
                throttle = 0.0
            hand_brake = False

            control = carla.VehicleControl()
            control.steer = float(steer)
            control.throttle = float(throttle)
            control.brake = float(brake)
            control.hand_brake = hand_brake

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
            direction = 6.
        elif direction == RoadOption.CHANGELANERIGHT:
            direction = 7.
        else:
            direction = 2.
        # print(direction)
        return direction
