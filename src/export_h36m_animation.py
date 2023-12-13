'''
applies a H3.6M motion-captured animation to an inferred pictorial figure
scales the H3.6M skeleton according to the pictorial figure
exports translations (i.e. x/y/z) and rotations (i.e. 2 rotation angles to canonical bone, 2 rotation angles to H3.6M bone)
'''

from data_utils import load_data
from config import H36M_NAMES_inv, POSE_POINTS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data_utils_mod import body_size
import json
import numpy as np
import math
from scipy.spatial.transform import Rotation
import os

def tx(point, matrix):
    [x, y, z] = point
    
    vector = (
        matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
        matrix[1][0] * x + matrix[1][1] * y + matrix[1][2] * z + matrix[1][3],
        matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3]
    )
    return vector

"""
a = np.array([3,  1 ,  2]) / np.linalg.norm(np.array([3,  1 ,  2]))
b = np.array([1 , -2, 4]) / np.linalg.norm(np.array([1 , -2, 4]))

rotation_zx_a = math.atan2(a[2], a[0])
rotation_zx_b = math.atan2(b[2], b[0])

rot1 = Rotation.from_rotvec(rotation_zx_b * np.array([0, 1, 0]))
b_rot = rot1.apply(b)

rot4 = Rotation.from_rotvec(rotation_zx_a * np.array([0, 1, 0]))
a_rot = rot4.apply(a)

rotation_xy_b = math.atan2(b_rot[0], b_rot[1])
rotation_xy_a = math.atan2(a_rot[0], a_rot[1])

rot2 = Rotation.from_rotvec(rotation_xy_b * np.array([0, 0, 1]))
rot3 = Rotation.from_rotvec(-rotation_xy_a * np.array([0, 0, 1]))
rot4 = Rotation.from_rotvec(-rotation_zx_a * np.array([0, 1, 0]))

b_rot = rot2.apply(b_rot)
b_rot = rot3.apply(b_rot)
b_rot = rot4.apply(b_rot)

R = rot4.as_dcm() @ rot3.as_dcm() @ rot2.as_dcm() @ rot1.as_dcm()

M = np.zeros((4, 4))
M[:3, :3] = R
M[3, 3] = 1

print(a, tx(b, M), b_rot)
"""


folder_name = r"E:\CNN\implicit_functions\characters\output\airbnb"

POSE_POINT_MAPPING = {
    'right_ankle': 'RFoot', 
    'right_knee': 'RKnee',
    'right_hip': 'RHip',
    'left_hip': 'LHip',
    'left_knee': 'LKnee',
    'left_ankle': 'LFoot',
    'pelvis': 'Hip',
    'root': 'Thorax',
    'neck': 'Neck/Nose',
    'head': 'Head',
    'right_wrist': 'RWrist',   
    'right_elbow': 'RElbow',
    'right_shoulder': 'RShoulder',
    'left_shoulder': 'LShoulder',
    'left_elbow': 'LElbow',
    'left_wrist': 'LWrist',
}

BONES = [
    ['head', 'root'],
    ['root', 'left_shoulder'],
    ['root', 'right_shoulder'],
    ['left_shoulder', 'left_elbow'],
    ['right_shoulder', 'right_elbow'],
    ['left_elbow', 'left_wrist'],
    ['right_elbow', 'right_wrist'],
    ['root', 'left_hip'],
    ['root', 'right_hip'],
    ['left_hip', 'left_knee'],
    ['right_hip', 'right_knee'],
    ['left_knee', 'left_ankle'],
    ['right_knee', 'right_ankle'],
]

BODY_PARTS = {
    "torso": ['right_shoulder', 'left_shoulder', 'left_hip', 'right_hip'], # 
    "head": [], # "head"
    "right_upper_arm": ['right_shoulder', 'right_elbow'],
    "right_lower_arm": ['right_elbow', 'right_wrist'],
    "right_hand": ['right_elbow', 'right_wrist'],
    "right_upper_leg": ['right_hip', 'right_knee'],
    "right_lower_leg": ['right_knee', 'right_ankle'],
    "right_foot": ['right_knee', 'right_ankle'],
    "left_upper_leg": ['left_hip', 'left_knee'],
    "left_lower_leg": ['left_knee', 'left_ankle'],
    "left_foot": ['left_knee', 'left_ankle'],
    "left_upper_arm": ['left_shoulder', 'left_elbow'],
    "left_lower_arm": ['left_elbow', 'left_wrist'],
    "left_hand": ['left_elbow', 'left_wrist']
}

BODY_PARTS_NARRAT3D = {
    "right_hand": ['right_wrist', 'right_middle1'],
    # "right_foot": ['right_ankle', 'right_foot'],
    # "left_foot": ['left_ankle', 'left_foot'],
    "left_hand": ['left_wrist', 'left_middle1']
}


with open(os.path.join(folder_name, "skeleton_front.json")) as file:
    pictorial_pose = json.load(file)

pictorial_pose = np.array(list(pictorial_pose.values()))


action = "Greeting"
subjects = [1]
N_JOINTS_H36M = 32
data = load_data("../data/h36m", subjects, [action], 3)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

for key, human_poses in data.items():
    if (key[2] == action + ".cdf"):
        skeleton_animation = []
        rotation_animation = []
        
        for frame_nr, human_pose in enumerate(human_poses):            
            human_pose = human_pose.reshape(-1, 3)
            
            xs = human_pose[:, 0] / 1000 / body_size * 600 + 300
            ys = 600 - human_pose[:, 2] / 1000 / body_size * 600
            zs = 300 - human_pose[:, 1] / 1000 / body_size * 600
            
            human_pose = np.stack([xs, ys, zs], axis=1)

            scaled_pose_points = {
                "head": human_pose[H36M_NAMES_inv["Head"]]
            }

            for bone_point_start, bone_point_end in BONES:
                pictorial_bone_start = pictorial_pose[POSE_POINTS[bone_point_start]]
                pictorial_bone_end = pictorial_pose[POSE_POINTS[bone_point_end]]
                pictorial_bone = pictorial_bone_end - pictorial_bone_start
                pictorial_bone_length = np.linalg.norm(pictorial_bone)
                
                human_bone_start = human_pose[H36M_NAMES_inv[POSE_POINT_MAPPING[bone_point_start]]]
                human_bone_end = human_pose[H36M_NAMES_inv[POSE_POINT_MAPPING[bone_point_end]]]
                human_bone = human_bone_end - human_bone_start
                human_bone_length = np.linalg.norm(human_bone)
                
                scaled_pose_point_start = scaled_pose_points[bone_point_start]
                scaling_factor = pictorial_bone_length / human_bone_length
                scaled_pose_point_end = scaled_pose_point_start + human_bone * scaling_factor
                scaled_pose_points[bone_point_end] = scaled_pose_point_end                

            if (frame_nr == 0):
                scaled_pose_points_list = np.array(list(scaled_pose_points.values()))
                
                xs = scaled_pose_points_list[:, 0]
                ys = scaled_pose_points_list[:, 1]
                zs = scaled_pose_points_list[:, 2]
    
                # xs = pictorial_pose[:, 0] 
                # ys = pictorial_pose[:, 1] 
                # zs = pictorial_pose[:, 2] 
    
                ax.scatter(xs, ys, zs, c="black")
                ax.set_xlim([0, 600])
                ax.set_ylim([0, 600])
                ax.set_zlim([0, 600])
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                
                # plt.show()
            
            rotations = []
            
            for body_part_name, keypoints in BODY_PARTS.items():
                if (keypoints == []):
                    rotations.append([0, 0, 0, 0])
                    continue            
                
                pictorial_keypoints = BODY_PARTS_NARRAT3D.get(body_part_name)
                
                if (pictorial_keypoints == None):
                    pictorial_keypoints = keypoints
                
                if (body_part_name == "torso"):
                    pictorial_bone_start1 = pictorial_pose[POSE_POINTS["right_shoulder"]]
                    pictorial_bone_start2 = pictorial_pose[POSE_POINTS["left_shoulder"]]
                    pictorial_bone_start = np.average([pictorial_bone_start1, pictorial_bone_start2], axis=0)
                    
                    pictorial_bone_end1 = pictorial_pose[POSE_POINTS["right_hip"]]
                    pictorial_bone_end2 = pictorial_pose[POSE_POINTS["left_hip"]]
                    pictorial_bone_end = np.average([pictorial_bone_end1, pictorial_bone_end2], axis=0)
                else :
                    bone_point_start, bone_point_end = pictorial_keypoints
                    pictorial_bone_start = pictorial_pose[POSE_POINTS[bone_point_start]]
                    pictorial_bone_end = pictorial_pose[POSE_POINTS[bone_point_end]]
                pictorial_bone = pictorial_bone_end - pictorial_bone_start 
                pictorial_bone_norm = pictorial_bone / np.linalg.norm(pictorial_bone)
                
                if (body_part_name == "torso"):
                    human_bone_start1 = scaled_pose_points["right_shoulder"]
                    human_bone_start2 = scaled_pose_points["left_shoulder"]
                    human_bone_start = np.average([human_bone_start1, human_bone_start2], axis=0)
                    
                    human_bone_end1 = scaled_pose_points["right_hip"]
                    human_bone_end2 = scaled_pose_points["left_hip"]
                    human_bone_end = np.average([human_bone_end1, human_bone_end2], axis=0)
                else :    
                    bone_point_start, bone_point_end = keypoints
                    human_bone_start = scaled_pose_points[bone_point_start]
                    human_bone_end = scaled_pose_points[bone_point_end]
                    
                human_bone = human_bone_end - human_bone_start
                human_bone_norm = human_bone / np.linalg.norm(human_bone)
                
                rotation_zx_human = math.atan2(human_bone[2], human_bone[0])
                rotation_zx_pictorial = math.atan2(pictorial_bone[2], pictorial_bone[0])
                
                rot1 = Rotation.from_rotvec(rotation_zx_pictorial * np.array([0, 1, 0]))
                pictorial_rot = rot1.apply(pictorial_bone_norm)
                
                rot4 = Rotation.from_rotvec(rotation_zx_human * np.array([0, 1, 0]))
                human_bone_rot = rot4.apply(human_bone)
                
                rotation_xy_pictorial = math.atan2(pictorial_rot[0], pictorial_rot[1])
                rotation_xy_human = math.atan2(human_bone_rot[0], human_bone_rot[1])
                
                rot2 = Rotation.from_rotvec(rotation_xy_pictorial * np.array([0, 0, 1]))
                rot3 = Rotation.from_rotvec(-rotation_xy_human * np.array([0, 0, 1]))
                rot4 = Rotation.from_rotvec(-rotation_zx_human * np.array([0, 1, 0]))
                
                pictorial_rot = rot2.apply(pictorial_rot)
                pictorial_rot = rot3.apply(pictorial_rot)
                pictorial_rot = rot4.apply(pictorial_rot)
                
                R = rot4.as_dcm() @ rot3.as_dcm() @ rot2.as_dcm() @ rot1.as_dcm()
                
                M = np.zeros((4, 4))
                M[:3, :3] = R
                M[3, 3] = 1
                
                """print(human_bone / np.linalg.norm(human_bone), 
                      pictorial_rot / np.linalg.norm(pictorial_rot),
                      tx(pictorial_bone, M) / np.linalg.norm(tx(pictorial_bone, M)))"""
                
                """print(body_part_name, 
                      rotation_zx_pictorial / math.pi * 180,
                      rotation_xy_pictorial / math.pi * 180,
                      rotation_xy_human / math.pi * 180,
                      rotation_zx_human / math.pi * 180)"""

                rotations.append([rotation_zx_pictorial, rotation_xy_pictorial, rotation_xy_human, rotation_zx_human]) # M.tolist()
            
            
            skeleton = [[0, 0, 0]] * 16
            
            for scaled_pose_point_name, scaled_pose_point in scaled_pose_points.items():
                keypoint_index = POSE_POINTS[scaled_pose_point_name]
                
                x = float(scaled_pose_point[0])
                y = float(scaled_pose_point[1])
                z = float(scaled_pose_point[2])
                
                skeleton[keypoint_index] = [x, y, z]
            
            skeleton_animation.append(skeleton)
            rotation_animation.append(rotations)
        
        np.save(os.path.join(folder_name, "skeleton_animation.npy"), skeleton_animation)
        np.save(os.path.join(folder_name, "rotation_animation.npy"), rotation_animation)
