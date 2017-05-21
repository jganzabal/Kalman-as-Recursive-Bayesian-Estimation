import numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Arc

def plot_robot_room(N = 21, loc = 0, doors = []):
    pos_radius = 0.8
    outer_raius = 1
    inner_big_radius = 0.95
    small_radius = 0.1*21.0/N
    angle = 0
    delta_angle = 2*np.pi/N
    fig, ax = plt.subplots(figsize=(8,8))
    #plt.figure(figsize=(5,5))
    im = plt.imread('robot.png')
    oi = OffsetImage(im, zoom = 0.25)
    rx = 0
    ry = 0
    
    for i in range(N):
        x = pos_radius*np.cos(angle)
        y = pos_radius*np.sin(angle)
        if (i+1) == loc:
            box = AnnotationBbox(oi, (x, y), frameon=False)
            ax.add_artist(box)
            pos_circle = plt.Circle((x, y), radius=small_radius, fill=True, color = 'y')
        else:
            pos_circle = plt.Circle((x, y), radius=small_radius, fill=False, color = 'g')
        plt.gca().add_patch(pos_circle)
        angle = angle + delta_angle
        plt.text(x, y, str(i+1), fontsize = 12, horizontalalignment='center', verticalalignment='center')
    circle_out = plt.Circle((0, 0), radius=outer_raius, fill=False)
    plt.gca().add_patch(circle_out)
    circle_in = plt.Circle((0, 0), radius=inner_big_radius, fill=False)
    plt.gca().add_patch(circle_in)
    circle_inner = plt.Circle((0, 0), radius=0.65, fill=False)
    plt.gca().add_patch(circle_inner)

    delta_angle_grad = delta_angle*180/np.pi
    for door in doors:
        door = Arc([0,0],inner_big_radius*2,inner_big_radius*2,angle=delta_angle_grad*door- delta_angle_grad*3/2,theta1=0, theta2=delta_angle_grad, color='y', linewidth='10')
        plt.gca().add_patch(door)

    plt.axis('scaled')
    plt.axis('off')
    plt.show()