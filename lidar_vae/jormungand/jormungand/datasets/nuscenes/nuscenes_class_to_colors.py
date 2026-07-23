# Color look-up table for NuScenes object categories.
# RGB tuples (0-255), designed for bright, vibrant identification.

NUSCENES_COLORS: dict[str, tuple[int, int, int]] = {
    # Vehicles
    "vehicle.car": (60, 180, 75),
    "vehicle.truck": (128, 0, 128),
    "vehicle.bus.rigid": (255, 150, 25),
    "vehicle.bus.bendy": (255, 200, 0),
    "vehicle.construction": (170, 110, 40),
    "vehicle.trailer": (100, 140, 255),
    "vehicle.motorcycle": (255, 10, 245),
    "vehicle.bicycle": (255, 255, 25),
    "vehicle.emergency.ambulance": (255, 50, 50),
    "vehicle.emergency.police": (0, 0, 200),
    # Pedestrians
    "human.pedestrian.adult": (220, 20, 60),
    "human.pedestrian.child": (255, 127, 80),
    "human.pedestrian.construction_worker": (255, 165, 0),
    "human.pedestrian.police_officer": (0, 100, 200),
    "human.pedestrian.wheelchair": (200, 130, 0),
    "human.pedestrian.stroller": (180, 100, 180),
    "human.pedestrian.personal_mobility": (150, 75, 0),
    # Animals
    "animal": (0, 200, 140),
    # Movable objects
    "movable_object.barrier": (128, 128, 0),
    "movable_object.trafficcone": (255, 80, 0),
    "movable_object.pushable_pullable": (180, 180, 180),
    "movable_object.debris": (120, 120, 80),
    # Static objects
    "static_object.bicycle_rack": (80, 80, 80),
}
