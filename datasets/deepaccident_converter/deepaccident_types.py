"""Type mappings for DeepAccident (CARLA) → UniTraj/MetaDrive ScenarioNet format."""

# CARLA label type strings → MetaDrive object type strings
CARLA_TO_METADRIVE_OBJECT = {
    'car':        'VEHICLE',
    'truck':      'VEHICLE',
    'van':        'VEHICLE',
    'motorcycle': 'CYCLIST',
    'cyclist':    'CYCLIST',
    'pedestrian': 'PEDESTRIAN',
}

# CARLA OpenDRIVE lane type strings → MetaDrive type strings
LANE_TYPE_MAP = {
    'driving':   'LANE_SURFACE_STREET',
    'biking':    'LANE_BIKE_LANE',
    'sidewalk':  'LANE_SURFACE_STREET',  # treat sidewalk as surface street
    'border':    'ROAD_EDGE_BOUNDARY',
    'shoulder':  'ROAD_EDGE_BOUNDARY',
    'parking':   'LANE_SURFACE_STREET',
    'stop':      'LANE_SURFACE_STREET',
    'none':      'ROAD_EDGE_MEDIAN',
}

# OpenDRIVE road mark type/weight/color → MetaDrive road line type strings
CARLA_ROAD_MARK_MAP = {
    # (type, weight) → MetaDrive road line type
    ('none',):                     None,  # skip
    ('solid',):                    'ROAD_LINE_SOLID_SINGLE_WHITE',
    ('solid_solid',):              'ROAD_LINE_SOLID_DOUBLE_WHITE',
    ('broken',):                   'ROAD_LINE_BROKEN_SINGLE_WHITE',
    ('broken_broken',):            'ROAD_LINE_BROKEN_DOUBLE_YELLOW',
    ('solid_broken',):             'ROAD_LINE_SOLID_SINGLE_YELLOW',  # approximate
    ('broken_solid',):             'ROAD_LINE_BROKEN_SINGLE_YELLOW',  # approximate
    ('botts dots',):              None,  # skip dots
    ('curb',):                    'ROAD_EDGE_BOUNDARY',
    ('grass',):                   None,  # skip grass
}

# Color mapping for road marks (overrides above when combined)
ROAD_MARK_COLOR_MAP = {
    'standard': 'WHITE',
    'white':    'WHITE',
    'yellow':   'YELLOW',
    'blue':     'WHITE',   # rare, map to white
}

DATASET_VERSION = 'v1.0'
AGENT_FILTER_MIN_VALID_FRAMES = 5

# Coordinate system: CARLA uses right-handed (x-forward, y-right, z-up)
COORDINATE = 'right-handed'

# Default frame rate for DeepAccident: 10Hz (scenarios are recorded at 10 fps)
FRAME_RATE_HZ = 10.0
