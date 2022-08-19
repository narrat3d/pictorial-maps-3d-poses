'''
helper methods for debugging data_utils
'''
from config import H36M_NAMES
import matplotlib.pyplot as plt
import numpy as np

def show_2d_pose(test_set):
    import viz
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    for poses in test_set.values():
        for pose in [poses[0]]:
            viz.show2Dpose(pose, ax, add_labels=True)
            ax.invert_yaxis()
        break
    
    plt.show()
    

def show_3d_pose(test_set):
    import viz
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
      
    for poses in test_set.values():
        for pose in [poses[0]]:
            viz.show3Dpose(pose, ax, lcolor="#9b59b6", rcolor="#2ecc71", add_labels=False)
        
        break
      
    plt.show()
    
 
def show_animated_3d_poses(poses):
    joints = np.array(H36M_NAMES) != ""
    
    import matplotlib.pyplot as plt
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter([], [], [])
    
    plt.draw()
    
    for i in range(poses.shape[0]):
        plt.cla()
        pose = np.reshape(poses[i], (-1, 3))
        xs = pose[:, 0]
        ys = pose[:, 1]
        zs = pose[:, 2]
        ax.scatter(xs[joints], ys[joints], zs[joints], c="black")
        ax.set_xlim([-2000, 2000])
        ax.set_ylim([-2000, 2000])
        ax.set_zlim([0, 2000])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        fig.canvas.draw_idle()
        plt.pause(0.001)
    
    plt.waitforbuttonpress()