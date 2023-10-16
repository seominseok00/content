from typing import Dict, Tuple, Text

import numpy as np
import pprint

from highway_env import utils
from highway_env.envs.common.abstract import AbstractEnv, MultiAgentWrapper
from highway_env.road.lane import LineType, StraightLane, SineLane, CircularLane, AbstractLane
from highway_env.road.regulation import RegulatedRoad
from highway_env.road.road import RoadNetwork
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Obstacle
from highway_env.vehicle.controller import ControlledVehicle, MDPVehicle

class IntersectionMergeEnv(AbstractEnv):

    ACTIONS: Dict[int, str] = {
        0: 'SLOWER',
        1: 'IDLE',
        2: 'FASTER'
    }
    ACTIONS_INDEXES = {v: k for k, v in ACTIONS.items()}

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 5,
                "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
                "absolute": True,
                "normalize": False,
                "order": "sorted",
                # "order": "shuffled",
                "observe_intentions": False  # Observe the destinations of other vehicles
            },
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": False,
                "target_speeds": [0, 7.5, 15]
            },
            "duration": 60,  # [s]
            "destination": "d",
            "ego_lane": ("o2", "ir2", 0),
            "controlled_vehicles": 1,
            "initial_vehicle_count": 5,
            "spawn_probability": 0.6,
            "screen_width": 600,
            "screen_height": 600,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5 * 1.3,
            "normalize_reward": False,
            "offroad_terminal": False
        })
        return config


    def _reward(self, action: int) -> float:
        """Per-agent reward signal."""
        rewards = self._rewards(action)
        # reward = sum(self.config.get(name, 0) * reward for name, reward in rewards.items())
        reward = 0

        for key in rewards:
            if key == 'collision_reward':  # 차량 충돌 발생시 -1
                reward -= rewards[key]
            elif key == 'speed_reward':  # 속도에 따라 0 ~ 1
                reward += rewards[key]
            elif key.startswith('expect_reward'):  # -0.5, -0.25, ···
                reward -= rewards[key]
                
        reward *= rewards["on_road_reward"]  # 차량이 차선을 유지하지 않을 경우 False가 돼서 reward가 0이 됨
        if self.config["normalize_reward"]:
            reward = utils.lmap(reward, [self.config["collision_reward"], self.config["speed_reward"]], [0, 1])
        
        return reward

    def _rewards(self, action: int) -> Dict[Text, float]:
        """Per-agent per-objective reward signal."""
        rewards = {}

        ego_vehicle = self.ego_vehicle

        ego_speed = ego_vehicle.speed
        speed_reward = (15 - np.sqrt((ego_vehicle.speed - 15)**2)) / 15

        ego_positions  = self._predict_position_constant_speed(self.ego_vehicle, 3)

        reward_idx = 1
        for vehicle in self.road.vehicles:
            if not isinstance(vehicle, MDPVehicle):  # ego_vehicle이 아닌 차량만 확인
                other_positions = self._predict_position_constant_speed(vehicle, 3)  # 차량의 경로 생성 (3 step)
                for idx, (ego, other) in enumerate(zip(ego_positions, other_positions)):
                    if self._precheck_collision(ego, other, ego_speed):
                        collision_idx = (idx // 15) + 1  # index를 1부터 시작
                        key = f"expect_reward{reward_idx}"
                        rewards[key] = 1 / (collision_idx * 2)  # 2를 곱함으로써 0.5, 0.25부터 시작
                        reward_idx += 1
                        break

        rewards['collision_reward'] = self.vehicle.crashed
        rewards['speed_reward'] = speed_reward if ego_speed >= 5 and ego_speed <= 15 else 0
        rewards['on_road_reward'] = ego_vehicle.on_road
        
        return rewards

    def _find_nearest_vehicles(self) -> Tuple[Vehicle, Vehicle, Vehicle]:
        intersection_roads = [("o2", "ir2", 0), ("ir2", " il0", 0), ("il0", "o0", 0), \
                             ("ir2", "il1", 0), ("il1", "o1", 0), \
                             ("ir2", "il3", 0), ("il3", "o3", 0), \
                             ("o0", "ir0", 0), ("ir0", "il2", 0), ("il2", "o2", 0), \
                             ("ir0", "il3", 0), ("il3", "o3", 0), \
                             ("ir0", "il1", 0), ("il1", "o1", 0), \
                             ("o1", "ir1", 0), ("ir1," "il3", 0), ("il3", "o3", 0), \
                             ("ir1", "il0", 0), ("il0", "o0", 0), \
                             ("ir1", "il2", 0), ("il2", "o2", 0), \
                             ("o3", "ir3", 0), ("ir3", "il1", 0), ("il1", "o1", 0), \
                             ("ir3", "il2", 0), ("il2", "o2", 0), \
                             ("ir3", "il0", 0), ("il0", "o0", 0)]
        
        merging_roads = [("o3", "b", 0), ("b", "c", 0), ("c", "d", 0), \
                        ("j", "k", 0), ("k", "b", 0), ("b", "c", 1)]

        ego_vehicle = self.ego_vehicle

        near_vehicles = []

        vehicles = [vehicle for vehicle in self.road.vehicles if not isinstance(vehicle, MDPVehicle)]

        for vehicle in vehicles:
            if ego_vehicle.lane_index in intersection_roads:
                if vehicle.lane_index in intersection_roads:
                    near_vehicles.append(vehicle)
            elif ego_vehicle.lane_index in merging_roads:
                if vehicle.lane_index in merging_roads:
                    near_vehicles.append(vehicle)

        near_vehicles.sort(key=lambda vehicle: np.linalg.norm(ego_vehicle.position - vehicle.position))
       
        while len(near_vehicles) < 2:
            near_vehicles.append(None)

        return near_vehicles[:2]
            
    def _is_terminated(self) -> bool:
        return any(vehicle.crashed for vehicle in self.controlled_vehicles) \
               or all(self.has_arrived(vehicle) for vehicle in self.controlled_vehicles) \
               or (self.config["offroad_terminal"] and not self.vehicle.on_road)

    def _agent_is_terminal(self, vehicle: Vehicle) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return (vehicle.crashed or
                self.has_arrived(vehicle))

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return self.time >= self.config["duration"]

    '''
    def _info(self, obs: np.ndarray, action: int) -> dict:
        info = super()._info(obs, action)
        info["agents_rewards"] = tuple(self._agent_reward(action, vehicle) for vehicle in self.controlled_vehicles)
        info["agents_dones"] = tuple(self._agent_is_terminal(vehicle) for vehicle in self.controlled_vehicles)
        return info
    '''

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles(self.config["initial_vehicle_count"])

    def _predict_position_constant_speed(self, vehicle, total_time):
        dt = 1 / self.config["simulation_frequency"]
        LENGTH = 5

        position = vehicle.position
        speed = vehicle.speed
        heading = vehicle.heading

        positions = [vehicle.position]

        for _ in range(int(total_time / dt)):
            delta_f = vehicle.action['steering']
            beta = np.arctan(1 / 2 * np.tan(delta_f))
            speed_vector = speed * np.array([np.cos(heading + beta), np.sin(heading + beta)])
            position = positions[-1] + speed_vector * dt
            positions.append(position)

            # heading = speed * np.sin(beta) / (LENGTH / 2) * dt

        return positions[1:]
    
    def _precheck_collision(self, ego, other, speed):
        dt = 1 / self.config["simulation_frequency"]
        
        diagonal = 5.385164807134504
    
        if np.linalg.norm(other - ego) > (diagonal + diagonal) / 2 + speed * dt:
            return False
        else:
            return True

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = super().step(action)

        self._clear_vehicles()
        self._spawn_vehicle(spawn_probability=self.config["spawn_probability"])

        ego_positions  = self._predict_position_constant_speed(self.ego_vehicle, 1)

        return obs, reward, terminated, truncated, info

    def _make_road(self) -> None:
        """
        Make an 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 3 for horizontal straight lanes and right-turns
            - 1 for vertical straight lanes and right-turns
            - 2 for horizontal left-turns
            - 0 for vertical left-turns

        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # Incoming
            start = rotation @ np.array([lane_width / 2, access_length + outer_distance])
            end = rotation @ np.array([lane_width / 2, outer_distance])
            net.add_lane("o" + str(corner), "ir" + str(corner),
                         StraightLane(start, end, line_types=[s, c], priority=priority, speed_limit=10))
            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))
            net.add_lane("ir" + str(corner), "il" + str((corner - 1) % 4),
                         CircularLane(r_center, right_turn_radius, angle + np.radians(180), angle + np.radians(270),
                                      line_types=[n, c], priority=priority, speed_limit=10))
            # Left turn
            l_center = rotation @ (np.array([-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2]))
            net.add_lane("ir" + str(corner), "il" + str((corner + 1) % 4),
                         CircularLane(l_center, left_turn_radius, angle + np.radians(0), angle + np.radians(-90),
                                      clockwise=False, line_types=[n, n], priority=priority - 1, speed_limit=10))
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])
            net.add_lane("ir" + str(corner), "il" + str((corner + 2) % 4),
                         StraightLane(start, end, line_types=[s, n], priority=priority, speed_limit=10))
            # Exit
            start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)
            net.add_lane("il" + str((corner - 1) % 4), "o" + str((corner - 1) % 4),
                         StraightLane(end, start, line_types=[n, c], priority=priority, speed_limit=10))
            
        # Highway lanes
        ends = [150, 80, 50, 150]  # Before, converging, merge, after
        y = [-StraightLane.DEFAULT_WIDTH / 2, StraightLane.DEFAULT_WIDTH / 2]
        line_type = [[s, c], [n, c]]
        line_type_merge = [[s, c], [n, s]]
        
        i = 0
        net.add_lane("b1", "o3", StraightLane([110 + sum(ends[:2]), y[i]], [110, y[i]], line_types=line_type[i]))
        net.add_lane("c1", "b1", StraightLane([110 + sum(ends[:3]), y[i]], [110 + sum(ends[:2]), y[i]], line_types=line_type_merge[i]))
        net.add_lane("d1", "c1", StraightLane([110 + sum(ends), y[i]], [110 + sum(ends[:3]), y[i]], line_types=line_type[i]))

        i = 1
        net.add_lane("o3", "b", StraightLane([110, y[i]], [110 + sum(ends[:2]), y[i]], line_types=line_type[i]))
        net.add_lane("b", "c", StraightLane([110 + sum(ends[:2]), y[i]], [110 + sum(ends[:3]), y[i]], line_types=line_type_merge[i]))
        net.add_lane("c", "d", StraightLane([110 + sum(ends[:3]), y[i]], [110 + sum(ends), y[i]], line_types=line_type[i]))

        # Merging lane
        amplitude = 3.25
        ljk = StraightLane([110, 6.5 + 4 + 2], [110 + ends[0], 6.5 + 4 + 2], line_types=[c, c], forbidden=True)
        lkb = SineLane(ljk.position(ends[0], -amplitude), ljk.position(sum(ends[:2]), -amplitude),
                       amplitude, 2 * np.pi / (2*ends[1]), np.pi / 2, line_types=[c, c], forbidden=True)
        lbc = StraightLane(lkb.position(ends[1], 0), lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=[n, c], forbidden=True)
        
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)

        road = RegulatedRoad(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, net.get_lane(("b", "c", 1)).position(ends[2], 0)))     

        self.road = road

    def _make_vehicles(self, n_vehicles: int = 10) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        # Configure vehicles
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type.DISTANCE_WANTED = 10  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3
        for t in range(n_vehicles - 1):
            self._spawn_vehicle(np.linspace(0, 80, n_vehicles)[t])
        for _ in range(simulation_steps):
            [(self.road.act(), self.road.step(1 / self.config["simulation_frequency"])) for _ in range(self.config["simulation_frequency"])]

        # Challenger vehicle
        self._spawn_vehicle(60, spawn_probability=1, go_straight=True, position_deviation=0.1, speed_deviation=0)

        # Controlled vehicles
        self.controlled_vehicles = []
        for ego_id in range(0, self.config["controlled_vehicles"]):
            # ego_lane = self.road.network.get_lane(("o{}".format(ego_id % 4), "ir{}".format(ego_id % 4), 0))
            # ego_lane = self.road.network.get_lane(("o3", "il2", 0))
            ego_lane = self.road.network.get_lane(self.config["ego_lane"])
            destination = self.config["destination"] or "o" + str(self.np_random.randint(1, 4))
            self.ego_vehicle = self.action_type.vehicle_class(
                             self.road,
                             ego_lane.position(60 + 5*self.np_random.normal(1), 0),
                             speed=ego_lane.speed_limit,
                             heading=ego_lane.heading_at(60))
            try:
                self.ego_vehicle.plan_route_to(destination)
                self.ego_vehicle.speed_index = self.ego_vehicle.speed_to_index(ego_lane.speed_limit)
                self.ego_vehicle.target_speed = self.ego_vehicle.index_to_speed(self.ego_vehicle.speed_index)
            except AttributeError:
                pass

            self.road.vehicles.append(self.ego_vehicle)
            self.controlled_vehicles.append(self.ego_vehicle)
            for v in self.road.vehicles:  # Prevent early collisions
                if v is not self.ego_vehicle and np.linalg.norm(v.position - self.ego_vehicle.position) < 20:
                    self.road.vehicles.remove(v)

    def _spawn_vehicle(self,
                       longitudinal: float = 0,
                       position_deviation: float = 1.,
                       speed_deviation: float = 1.,
                       spawn_probability: float = 0.6,
                       go_straight: bool = False) -> None:
        if self.np_random.uniform() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])

        if route[0] != 3:
            vehicle = vehicle_type.make_on_lane(self.road, ("o" + str(route[0]), "ir" + str(route[0]), 0),
                                                longitudinal=(longitudinal + 5
                                                            + self.np_random.normal() * position_deviation),
                                                speed=8 + self.np_random.normal() * speed_deviation)
            
        else:
            if self.np_random.uniform() > spawn_probability:
                vehicle = vehicle_type.make_on_lane(self.road, ("j", "k", 0),
                                                    longitudinal=(longitudinal + 5
                                                                + self.np_random.normal() * position_deviation),
                                                    speed=8 + self.np_random.normal() * speed_deviation)

            else:
                vehicle = vehicle_type.make_on_lane(self.road, ("d1", "c1", 0),
                                                    longitudinal=(longitudinal + 5
                                                                + self.np_random.normal() * position_deviation),
                                                    speed=8 + self.np_random.normal() * speed_deviation)                
                
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        # 차량이 il ~ o로 가는 lane에 있고 해당 lane의 끝지점을 지나고 있으면 해당 차량을 제거
        is_leaving = lambda vehicle: "il" in vehicle.lane_index[0] and "o" in vehicle.lane_index[1] \
                                     and vehicle.lane.local_coordinates(vehicle.position)[0] >= vehicle.lane.length - 4 * vehicle.LENGTH \
                                     and not (vehicle.lane_index[0] == 'il3' and vehicle.lane_index[1] == 'o3') \
                                     and (vehicle.lane_index[0] == 'c' and vehicle.lane_index[1] == 'd' and vehicle.lane_index[3] == 0)
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle in self.controlled_vehicles or not (is_leaving(vehicle) or vehicle.route is None)]
    
    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 70) -> bool:
        return "c" in vehicle.lane_index[0] \
               and "d" in vehicle.lane_index[1] \
               and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance