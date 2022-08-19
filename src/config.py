import numpy as np

TRAINVAL_DATA_FOLDER = r"C:\Users\raimund\Downloads\tmp2"

NUM_CAMERAS = 4

class Config(): 
    MODEL_NAME = None
    ACTION = None
    USE_ORIGINAL_DATA = None
    USE_ORIGINAL_JOINTS = None
    USE_ORIGINAL_VALIDATION_DATA = None
    USE_OWN_VALIDATION_DATA = None
    CAMERA_FRAME = None

config = Config()

def load_config(name):
    config.MODEL_NAME = name
    
    load_config_method(name)
    derive_variables()

def load_config_method(name):
    config_method = CONFIGS[name]
    config_method()

def h36m_16j():
    config.ACTION = "All" 
    config.USE_ORIGINAL_DATA = True
    config.USE_ORIGINAL_VALIDATION_DATA = True
    config.USE_ORIGINAL_JOINTS = True
    config.USE_OWN_TEST_DATA = False
    config.CAMERA_FRAME = False

def h36m_16j_debug():
    load_config_method("h36m_16j")
    config.ACTION = "Walking"

def h36m_16j_cam():
    load_config_method("h36m_16j")
    config.CAMERA_FRAME = True

def h36m_16j_cam_debug():
    load_config_method("h36m_16j_cam")
    config.ACTION = "Walking"

def h36m_16j_inf_narrat3d_val():
    load_config_method("h36m_16j")
    config.MODEL_NAME = "h36m_16j"
    config.USE_ORIGINAL_VALIDATION_DATA = False
    
def h36m_16j_cam_inf_narrat3d_val():
    load_config_method("h36m_16j_cam")
    config.MODEL_NAME = "h36m_16j_cam"
    config.USE_ORIGINAL_VALIDATION_DATA = False

def narrat3d_16j():
    config.ACTION = "Directions"
    config.USE_ORIGINAL_DATA = False
    config.USE_ORIGINAL_VALIDATION_DATA = False
    config.USE_ORIGINAL_JOINTS = True
    config.USE_OWN_TEST_DATA = False
    config.CAMERA_FRAME = False    
    
def narrat3d_16j_cam():
    load_config_method("narrat3d_16j")
    config.CAMERA_FRAME = True   

def narrat3d_21j():
    config.ACTION = "Directions"
    config.USE_ORIGINAL_DATA = False
    config.USE_ORIGINAL_VALIDATION_DATA = False
    config.USE_ORIGINAL_JOINTS = False
    config.USE_OWN_TEST_DATA = False
    config.CAMERA_FRAME = False    

def narrat3d_21j_cam():
    load_config_method("narrat3d_21j")
    config.CAMERA_FRAME = True 

def narrat3d_21j_inf_narrat3d_test():
    load_config_method("narrat3d_21j")
    config.MODEL_NAME = "narrat3d_21j"
    config.USE_OWN_TEST_DATA = True

def derive_variables():
    if config.USE_ORIGINAL_JOINTS:
        config.NUM_JOINTS_WITH_ROOT = 16
        config.NUM_JOINTS = 16 # since the root joint is included in the original model
        config.ROOT_INDEX = 0
    else :
        config.NUM_JOINTS_WITH_ROOT = 21
        config.NUM_JOINTS = 20
        config.ROOT_INDEX = 6

CONFIGS = {
    "h36m_16j_debug": h36m_16j_debug,
    "h36m_16j_cam_debug": h36m_16j_cam_debug,
    "h36m_16j": h36m_16j,
    "h36m_16j_cam": h36m_16j_cam,
    "h36m_16j_inf_narrat3d_val": h36m_16j_inf_narrat3d_val,
    "h36m_16j_cam_inf_narrat3d_val": h36m_16j_cam_inf_narrat3d_val,
    "narrat3d_16j": narrat3d_16j,
    "narrat3d_16j_cam": narrat3d_16j_cam,
    "narrat3d_21j": narrat3d_21j,
    "narrat3d_21j_cam": narrat3d_21j_cam,
    "narrat3d_21j_inf_narrat3d_test": narrat3d_21j_inf_narrat3d_test
}

POSE_POINTS_SMPLX = {
    "root": 0,
    "pelvis": 1,
    "left_hip": 2,
    "left_knee": 3,
    "left_ankle": 4,
    "left_foot": 5,
    "right_hip": 6,
    "right_knee": 7,
    "right_ankle": 8,
    "right_foot": 9,
    "spine1": 10,
    "spine2": 11,
    "spine3": 12,
    "neck": 13,
    "head": 14,
    "jaw": 15,
    "left_eye_smplhf": 16,
    "right_eye_smplhf": 17,
    "left_collar": 18,
    "left_shoulder": 19,
    "left_elbow": 20,
    "left_wrist": 21,
    "left_index1": 22,
    "left_index2": 23,
    "left_index3": 24,
    "left_middle1": 25,
    "left_middle2": 26,
    "left_middle3": 27,
    "left_pinky1": 28,
    "left_pinky2": 29,
    "left_pinky3": 30,
    "left_ring1": 31,
    "left_ring2": 32,
    "left_ring3": 33,
    "left_thumb1": 34,
    "left_thumb2": 35,
    "left_thumb3": 36,
    "right_collar": 37,
    "right_shoulder": 38,
    "right_elbow": 39,
    "right_wrist": 40,
    "right_index1": 41,
    "right_index2": 42,
    "right_index3": 43,
    "right_middle1": 44,
    "right_middle2": 45,
    "right_middle3": 46,
    "right_pinky1": 47,
    "right_pinky2": 48,
    "right_pinky3": 49,
    "right_ring1": 50,
    "right_ring2": 51,
    "right_ring3": 52,
    "right_thumb1": 53,
    "right_thumb2": 54,
    "right_thumb3": 55
}

POSE_POINTS_SMPLX_INV = {str(v): k for k, v in POSE_POINTS_SMPLX.items()}

POSE_POINTS = {
    'right_ankle': 0, 
    'right_knee': 1,
    'right_hip': 2,
    'left_hip': 3,
    'left_knee': 4,
    'left_ankle': 5,
    'pelvis': 6, # hip
    'root': 7, # thorax
    'neck': 8,
    'head': 9,
    'right_wrist': 10,   
    'right_elbow': 11,
    'right_shoulder': 12,
    'left_shoulder': 13,
    'left_elbow': 14,
    'left_wrist': 15,
    'right_foot': 16,
    'left_foot': 17,
    'right_middle1': 18,
    'left_middle1': 19,
    'eyes': 20,
    'right_eye_smplhf': 21,
    'left_eye_smplhf': 22,
}

RIGHT_BONES = [
    ["right_foot", "right_ankle"],
    ["right_ankle", "right_knee"],
    ["right_knee", "right_hip"],
    ["right_hip", "pelvis"],
    ["right_middle1", "right_wrist"],
    ["right_wrist", "right_elbow"],
    ["right_elbow", "right_shoulder"],
    ["right_shoulder", "neck"]
]

LEFT_BONES = [
    ["left_foot", "left_ankle"],
    ["left_ankle", "left_knee"],
    ["left_knee", "left_hip"],
    ["left_hip", "pelvis"],
    ["left_middle1", "left_wrist"],
    ["left_wrist", "left_elbow"],
    ["left_elbow", "left_shoulder"],
    ["left_shoulder", "neck"],
    ["neck", "head"],
    ["head", "eyes"],
    ["neck", "root"],
    ["root", "pelvis"]
]

I = []
J = []
LR = []

for [joint1, joint2] in RIGHT_BONES:
    I.append(POSE_POINTS[joint1])
    J.append(POSE_POINTS[joint2])
    LR.append(0)

for [joint1, joint2] in LEFT_BONES:
    I.append(POSE_POINTS[joint1])
    J.append(POSE_POINTS[joint2])
    LR.append(1)

I = np.array(I, dtype=np.int32)
J = np.array(J, dtype=np.int32)
LR = np.array(LR, dtype=np.bool)

# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
H36M_NAMES = ['']*32
H36M_NAMES[0]  = 'Hip'
H36M_NAMES[1]  = 'RHip'
H36M_NAMES[2]  = 'RKnee'
H36M_NAMES[3]  = 'RFoot'
H36M_NAMES[6]  = 'LHip'
H36M_NAMES[7]  = 'LKnee'
H36M_NAMES[8]  = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

# Stacked Hourglass produces 16 joints. These are the names.
SH_NAMES = ['']*16
SH_NAMES[0]  = 'RFoot'
SH_NAMES[1]  = 'RKnee'
SH_NAMES[2]  = 'RHip'
SH_NAMES[3]  = 'LHip'
SH_NAMES[4]  = 'LKnee'
SH_NAMES[5]  = 'LFoot'
SH_NAMES[6]  = 'Hip'
SH_NAMES[7]  = 'Spine'
SH_NAMES[8]  = 'Thorax'
SH_NAMES[9]  = 'Head'
SH_NAMES[10] = 'RWrist'
SH_NAMES[11] = 'RElbow'
SH_NAMES[12] = 'RShoulder'
SH_NAMES[13] = 'LShoulder'
SH_NAMES[14] = 'LElbow'
SH_NAMES[15] = 'LWrist'


H36M_NAMES_inv = {name: i for [i, name] in enumerate(H36M_NAMES)}
