import numpy as np
from matplotlib import pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Arc

def plot_robot_room(N = 21, loc = 0, doors = [], sample_stats = None, figsize=(8,8)):
    pos_radius = 0.8
    outer_raius = 1
    inner_big_radius = 0.95
    small_radius = 0.1*21.0/N
    angle = 0
    delta_angle = 2*np.pi/N
    fig, ax = plt.subplots(figsize=figsize)
    #plt.figure(figsize=(5,5))
    im = plt.imread("./images/robot.png")
    oi = OffsetImage(im, zoom = 0.25)
    rx = 0
    ry = 0
    
    for i in range(N):
        x = pos_radius*np.cos(angle)
        y = pos_radius*np.sin(angle)
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

    stats_radius = 1.1
    if sample_stats is not None:
        max_door = 0
        max_hist = 0
        max_spread_hist = 0
        for sample_stat in sample_stats:
            if sample_stat is not None:
                if 'door' in sample_stat:
                    if max_door<sample_stat['door']:
                        max_door = sample_stat['door']
                if not set(range(N+1)).isdisjoint(list(sample_stat.keys())):
                    maximun = max(list(sample_stat.values()))
                    spread = max(list(sample_stat.keys())) - min(list(sample_stat.keys()))
                    if max_hist<maximun:
                        max_hist = maximun
                    if max_spread_hist<spread:
                        max_spread_hist = spread
        
        for i, sample_stat in enumerate(sample_stats):
            if sample_stat is not None:
                if 'door' in sample_stat:
                    prediction_arc = Arc([0,0],stats_radius*2,stats_radius*2,angle=(delta_angle_grad*(i+1) - delta_angle_grad*3/2),
                                        theta1=0, theta2=delta_angle_grad, color='g', linewidth=20*sample_stat['door']/max_door)
                    plt.gca().add_patch(prediction_arc)
                if not set(range(N+1)).isdisjoint(list(sample_stat.keys())):
                    angle_width = delta_angle_grad/(N+2)
                    angle_base_pos = (delta_angle_grad*(i+1) - delta_angle_grad*3/2)
                    prediction_arc = Arc([0,0],stats_radius*2,stats_radius*2,angle=angle_base_pos,
                                        theta1=0, theta2=angle_width/4  , color='k', linewidth=40.0)
                    plt.gca().add_patch(prediction_arc)
                    for key, value in sample_stat.items():
                        angle_position = angle_base_pos + (key+1)*angle_width

                        prediction_arc = Arc([0,0],stats_radius*2,stats_radius*2,angle=0,
                                        theta1=angle_position-angle_width, theta2=angle_position, color='g', linewidth=20.0*value/max_hist)
                        plt.gca().add_patch(prediction_arc)
    
    # Draw robot
    x = pos_radius*np.cos(delta_angle*(loc-1))
    y = pos_radius*np.sin(delta_angle*(loc-1))
    box = AnnotationBbox(oi, (x, y), frameon=False)
    ax.add_artist(box)
    pos_circle = plt.Circle((x, y), radius=small_radius, fill=True, color = 'y')
    plt.gca().add_patch(pos_circle)

    plt.axis('scaled')
    plt.axis('off')
    plt.show()