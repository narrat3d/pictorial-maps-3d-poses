'''
loads our own training, validation and test data.
stores the results after evaluation
'''
import os
import json
import numpy as np
import copy
from config import config, POSE_POINTS_SMPLX_INV, POSE_POINTS, SH_NAMES,\
    H36M_NAMES_inv, H36M_NAMES, TRAINVAL_DATA_FOLDER


current_folder = os.path.dirname(__file__)
body_size = 1.79 # = average size of H3.6M scanned body meshes for validation subjects (i.e., S9 and S11)

train_file_paths = []
num_train_files = 0
validation_file_paths = []
num_validation_files = 0
test_file_paths = []
num_test_files = 0

def initialize_trainval_data():
    global train_file_paths
    global num_train_files
    global validation_file_paths
    global num_validation_files

    file_paths = []
    
    trainval_subfolders = os.listdir(TRAINVAL_DATA_FOLDER)
    
    for sub_folder in trainval_subfolders:
        if (os.path.isfile(os.path.join(TRAINVAL_DATA_FOLDER, sub_folder))):
            continue
        
        for view in ["front", "left", "back", "right"]:
            subject_path = os.path.join(TRAINVAL_DATA_FOLDER, sub_folder, "skeleton_%s.json" % view)
            file_paths.append(subject_path)
    
    num_files = len(file_paths)
    num_train_files = round(num_files * 0.9)
    num_validation_files = num_files - num_train_files
    train_file_paths = file_paths[:num_train_files]
    validation_file_paths = file_paths[num_train_files:]


def set_test_folders(data_folder, sub_folders):
    global test_file_paths
    global num_test_files
    
    sub_folders_names = os.listdir(data_folder) if sub_folders == "" else sub_folders.split("|")

    for sub_folder in sub_folders_names:
        if (os.path.isfile(os.path.join(data_folder, sub_folder))):
            continue
        
        subject_path = os.path.join(data_folder, sub_folder, "skeleton_2d.json")
        test_file_paths.append(subject_path)
        
    num_test_files = len(test_file_paths)


def set_model_folder(model_folder_):
    global model_folder
    model_folder = model_folder_
    
    
def get_validation_or_test_data():
    if (not config.USE_OWN_TEST_DATA):
        file_paths = validation_file_paths
        num_files = num_validation_files
    else :
        file_paths = test_file_paths
        num_files = num_test_files
        
    return file_paths, num_files


def load_data(data_dir, test_subjects, actions, dim):
    #train data
    if (len(test_subjects) == 5):
        file_paths = train_file_paths
        num_files = num_train_files
        
    # test data
    if (len(test_subjects) == 2):
        (file_paths, num_files) = get_validation_or_test_data()
        
    result_array = np.zeros((num_files, 32*dim))
    
    
    for i, file_path in enumerate(file_paths): 
        with open(file_path) as file:
            skeleton = json.load(file)
            
        joint_array = np.zeros((32, dim))
        
        for joint_index, coords in skeleton.items():
            if not config.USE_OWN_TEST_DATA or len(test_subjects) == 5:    
                smplx_joint_name = POSE_POINTS_SMPLX_INV[joint_index]
                joint = POSE_POINTS.get(smplx_joint_name)
            else :
                joint = int(joint_index)
            
            if joint is None or (config.USE_ORIGINAL_JOINTS and joint > 15):
                continue
            
            if config.USE_ORIGINAL_JOINTS:                
                joint_name = SH_NAMES[joint]
                joint_index = H36M_NAMES_inv[joint_name]
            else :
                joint_index = joint
                
                if (joint == POSE_POINTS["eyes"] and not config.USE_OWN_TEST_DATA):
                    # eyes point is present for inference
                    continue
            
            x = (coords[0] - 300) / 600 * body_size * 1000
            if not config.USE_OWN_TEST_DATA:
                y = (300 - coords[2]) / 600 * body_size * 1000
            else :
                y = 0
            z = (600 - coords[1]) / 600 * body_size * 1000
            
            joint_array[joint_index] = np.array([x, y, z])
  

        if config.USE_ORIGINAL_JOINTS: # calculate H36M "neck/nose" for visualization purposes
            joint_array[14] = (joint_array[13] + joint_array[15]) / 2
        elif not config.USE_OWN_TEST_DATA : # calculate middle of eyes from left and right eye
            joint_array[20] = (joint_array[21] + joint_array[22]) / 2
            joint_array[21] = np.zeros(3)
            joint_array[22] = np.zeros(3)
        
        joint_array_flattened = joint_array.flatten()
        result_array[i,] = joint_array_flattened
    
    dataset_name = (9, "Directions", "Directions.cdf")

    return {dataset_name: result_array}


def store_results(results_3d, dims_to_use):
    if config.USE_ORIGINAL_DATA:
        print("Can only store results for own data")
        return
    
    root_index = config.ROOT_INDEX
    num_joints = config.NUM_JOINTS_WITH_ROOT
      
    if not config.USE_ORIGINAL_JOINTS:  
        hip_coords = [root_index*3, root_index*3+1, root_index*3+2]
        dims_to_use = np.insert(dims_to_use, root_index*3, hip_coords) 
      
    (file_paths, num_files) = get_validation_or_test_data()  
    
    if config.CAMERA_FRAME:
        # use results of first camera
        first_item = list(results_3d.items())[0]
        results_3d = {first_item[0]: first_item[1]}
    
    for data in results_3d.values():
        filtered_data = data["pred"][:, dims_to_use]
        filtered_data = filtered_data.reshape(num_files, num_joints, 3)
        
        skeleton = {}
        
        for index, file_path in enumerate(file_paths):
            joints = filtered_data[index]
            
            with open(file_path) as skeleton_2d_file:
                skeleton_2d = json.load(skeleton_2d_file)
            
            for joint_index in range(num_joints):
                coords = joints[joint_index]
                
				# take the original x and y coordinate instead of the predicted ones
                x = skeleton_2d[str(joint_index)][0] # coords[0] / 1000 / body_size * 600 + 300
                y = skeleton_2d[str(joint_index)][1] # 600 - coords[2] / 1000 / body_size * 600
                z = 300 - coords[1] / 1000 / body_size * 600 
                  
                skeleton[joint_index] = [x, y, z]
            
            if not config.USE_OWN_TEST_DATA:
                skeleton_path = file_path.replace(".json", "_pred.json")
            else :
                skeleton_path = file_path.replace("2d.json", "front.json")
            
            print (skeleton_path)
            
            with open(skeleton_path, "w") as file:
                json.dump(skeleton, file)
            
        break


def load_mean_and_std(dim):    
    if (dim == 2):
        data_mean = np.load(os.path.join(model_folder, "normalizations", config.MODEL_NAME, "data_mean_2d.npy"))
        data_std = np.load(os.path.join(model_folder, "normalizations", config.MODEL_NAME, "data_std_2d.npy"))  
    else :
        data_mean = np.load(os.path.join(model_folder, "normalizations", config.MODEL_NAME, "data_mean_3d.npy"))
        data_std = np.load(os.path.join(model_folder, "normalizations", config.MODEL_NAME, "data_std_3d.npy"))  
        
    return data_mean, data_std


def normalization_stats(complete_data, dim, predict_14=False ):
    """Computes normalization statistics: mean and stdev, dimensions used and ignored
    
    Args
      complete_data: nxd np array with poses
      dim. integer={2,3} dimensionality of the data
      predict_14. boolean. Whether to use only 14 joints
    Returns
      data_mean: np vector with the mean of the data
      data_std: np vector with the standard deviation of the data
      dimensions_to_ignore: list of dimensions not used in the model
      dimensions_to_use: list of dimensions used in the model
    """
    if not dim in [2,3]:
        raise ValueError('dim must be 2 or 3')
    
    if not config.USE_OWN_TEST_DATA: 
        data_mean = np.mean(complete_data, axis=0)
        data_std  =  np.std(complete_data, axis=0)
    else : # test data
        data_mean, data_std = load_mean_and_std(dim)
    
    root_index = config.ROOT_INDEX
    num_joints = config.NUM_JOINTS_WITH_ROOT
    
    if dim == 2:
        hip_coords = [root_index*2, root_index*2+1]
        point_coords = np.arange(2 * 32)
        dimensions_to_use = point_coords[:2 * num_joints]
        dimensions_to_use = np.delete(dimensions_to_use, hip_coords)
        dimensions_to_ignore = np.concatenate([hip_coords, point_coords[2 * num_joints:]])
    else: # dim == 3
        hip_coords = [root_index*3, root_index*3+1, root_index*3+2]
        point_coords = np.arange(3 * 32)
        dimensions_to_use = point_coords[:3 * num_joints]
        dimensions_to_use = np.delete(dimensions_to_use, hip_coords)
        dimensions_to_ignore = np.concatenate([hip_coords, point_coords[3 * num_joints:]])
    
    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use


def postprocess_3d( poses_set ):
    """Center 3d points around root
    
    Args
      poses_set: dictionary with 3d data
    Returns
      poses_set: dictionary with 3d data centred around root (center hip) joint
      root_positions: dictionary with the original 3d position of each pose
    """
    root_positions = {}
    root_index = config.ROOT_INDEX
    
    for k in poses_set.keys():
        # Keep track of the global position
        root_positions[k] = copy.deepcopy(poses_set[k][:,root_index*3:(root_index+1)*3])
        
        # Remove the root from the 3d position
        poses = poses_set[k]
        poses = poses - np.tile( poses[:,root_index*3:(root_index+1)*3], [1, len(H36M_NAMES)] )
        poses_set[k] = poses
    
    return poses_set, root_positions


def delete_depth_coordinate(poses3d):
    rows = np.arange(32)*3 + 1
    return np.delete(poses3d, rows, 1)


def project_to_cameras(poses_set, cams, ncams=4):
    for key, poses in poses_set.items():
        poses_set[key] = delete_depth_coordinate(poses)
        
    return poses_set


def smplx_to_narrat3d_skeleton(smplx_skeleton_path):
    narrat3d_skeleton = {}
    
    with open(smplx_skeleton_path) as smplx_skeleton_file:
        smplx_skeleton = json.load(smplx_skeleton_file)
        
    for pose_point_index, coords in smplx_skeleton.items():
        pose_point_name = POSE_POINTS_SMPLX_INV[pose_point_index]
        mapped_pose_point_index = POSE_POINTS.get(pose_point_name)
        
        if (mapped_pose_point_index is None):
            continue
        
        narrat3d_skeleton[mapped_pose_point_index] = coords
    
    # eyes
    narrat3d_skeleton[20] = list((np.array(narrat3d_skeleton[21]) + np.array(narrat3d_skeleton[22])) / 2)
    
    narrat3d_skeleton_path = smplx_skeleton_path.replace(".json", "_new.json")
    
    with open(narrat3d_skeleton_path, "w") as narrat3d_skeleton_file:
        json.dump(narrat3d_skeleton, narrat3d_skeleton_file)
        
    
if __name__ == '__main__':
    smplx_to_narrat3d_skeleton(r"E:\CNN\implicit_functions\smpl-x\rp_steve_posed_001_0_0_male_small\skeleton_front_original.json")