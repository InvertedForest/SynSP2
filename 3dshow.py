import matplotlib.pyplot as plt  
import numpy as np
import matplotlib.style as mplstyle
mplstyle.use('fast')


# place the pose data as a string here
coor0 = '''

tensor([ 0.3287,  0.7144, -0.0010,  0.2097,  0.3358, -0.0408,  0.0189, -0.0012,
        -0.0603, -0.0189,  0.0012,  0.0603, -0.0408,  0.3800,  0.0797, -0.0735,
         0.7767,  0.0539,  0.2283, -0.4945, -0.3446, -0.0180, -0.4845, -0.3920,
        -0.0195, -0.5487, -0.1504, -0.0440, -0.5345,  0.1708,  0.0230, -0.4423,
         0.4025,  0.2319, -0.4876,  0.2852, -0.0383, -0.6032,  0.0145, -0.0178,
        -0.8648,  0.0377], device='cuda:0')


'''

def parse_coor_string(coor0):
    coor0_left = 0
    coor0_right = 1
    bracket = 0
    enter =  0
    for i, c in enumerate(coor0):
        if c == '[':
            bracket += 1
            if enter == 0:
                enter = 1
                coor0_left = i
        if c == ']' and enter != 0:
            bracket -= 1
            if enter != 0 and bracket == 0:
                coor0_right = i
                break
    coor1 = eval(f'np.array({coor0[coor0_left:coor0_right+1]})')
    return coor1

coor1 = parse_coor_string(coor0)

def pre_process(coor):
    if len(coor.shape) != 2:
        coor = coor.reshape(-1, 3)
    if coor.shape[0] != 14:
        raise Exception

    coor[:, 1] = -coor[:, 1]
    coor = coor[:, [0, 2, 1]]
    return coor
    

AIST_VIBE_3D_EDGES = [  
    [0, 1, 0],  
    [1, 2, 0],  
    [2, 8, 0],  
    [3, 9, 1],  
    [4, 3, 1],  
    [5, 4, 1],  
    [6, 7, 0],  
    [7, 8, 0],  
    [8, 12, 0],  
    [9, 12, 1],  
    [10, 9, 1],  
    [11, 10, 1],  
    [12, 13, 1],  
]  
  
fig = plt.figure() 

def plot_3d_skeleton(coor, sub=111, edges=[edge[:2] for edge in AIST_VIBE_3D_EDGES], color=['red', 'black', 'blue']):  
    ax = fig.add_subplot(sub, projection='3d')  
    coor = pre_process(coor)
  

    for i, (x, y, z) in enumerate(coor):  
        ax.scatter(x, y, z, s=10, color='red')  
        ax.text(x, y, z, f'{i}', size=8, zorder=1, color='black')  
  

    for edge in edges:  
        start_id, end_id = edge  
        start_coor = coor[start_id]  
        end_coor = coor[end_id]  
        ax.plot([start_coor[0], end_coor[0]],  
                 [start_coor[1], end_coor[1]],  
                  [start_coor[2], end_coor[2]], color='blue', linewidth=2)  
  

    ax.set_xlabel('0')  
    ax.set_ylabel('2')  
    ax.set_zlabel('1')  
    ax.set_xlim(np.min(coor[:, 0]), np.max(coor[:, 0]))
    ax.set_ylim(np.min(coor[:, 1]), np.max(coor[:, 1]))
    ax.set_zlim(np.min(coor[:, 2]), np.max(coor[:, 2]))
    ax.set_aspect('equal', adjustable='box')
    ax.view_init(elev=0, azim=-90)
  

     
  


plot_3d_skeleton(coor1, sub=121)
# plot_3d_skeleton(coor2, sub=122)
# plot_3d_skeleton(coor3, sub=133)


plt.show()  