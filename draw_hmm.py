from matplotlib.patches import Ellipse, Arc, ConnectionPatch, ConnectionStyle, FancyArrow
from matplotlib import pyplot as plt
delta = 2
arrow_width = 0.03

def draw_state(state=0, state_label = 0):
    radius = 0.5
    x = state*delta
    y = 0
    pos_circle = plt.Circle((x, y), radius=radius, fill=False, color='b', lw=2)
    plt.gca().add_patch(pos_circle)
    plt.text(x, y, '$S_{%s}$'%(str(state_label)), fontsize=12, horizontalalignment='center', verticalalignment='center')


def draw_self_transition(state=0, text='0.55'):
    x_pos = state*delta
    y_pos = 0.7
    height = 0.60
    width = 0.55
    ellipse = Arc([x_pos,y_pos],width,height,angle=0,theta1=-50, theta2=230.0, color='k', linewidth='1')
    plt.gca().add_patch(ellipse)
    arrow2 = FancyArrow(x_pos-0.01, y_pos+height/2, 0.01, -0.002, width=arrow_width, length_includes_head=False, head_width=None, head_length=None, shape='full', overhang=0, head_starts_at_zero=False, color='k')
    plt.gca().add_patch(arrow2)
    plt.text(x_pos, y_pos + 1.1*height/2, text, fontsize = 12, horizontalalignment='center', verticalalignment='bottom')


def draw_transition(state1=0, state2=1, text=0.1):
    radius = 0.5
    x1 = state1*delta + radius
    diff_state = abs(state2-state1)
    x2 = delta*diff_state - 2*radius
    y = 0
    if diff_state == 1:
        ellipse = Arc([x1+x2/2,y],x2,0,angle = 0,theta1=0, theta2=180.0,color='k', linewidth='1')
        plt.text(x1+x2/2, y+ 0.1, text, fontsize = 12, horizontalalignment='center', verticalalignment='bottom')
        arrow2 = FancyArrow(x1+x2/2-0.01, 0, 0.01, 0, width=arrow_width, length_includes_head=False, head_width=None, head_length=None, shape='full', overhang=0, head_starts_at_zero=False, color='k')
    else:
        new_y = -0.2
        arc_rad = 3.3
        ellipse = Arc([x1+x2/2,new_y],x2*1.2,arc_rad,angle = 0,theta1=20, theta2=160.0,color='k', linewidth='1')
        plt.text(x1+x2/2, arc_rad/2+new_y+ 0.1, text, fontsize = 12, horizontalalignment='center', verticalalignment='bottom')
        arrow2 = FancyArrow(x1+x2/2-0.01, arc_rad/2+new_y, 0.01, 0, width=arrow_width, length_includes_head=False, head_width=None, head_length=None, shape='full', overhang=0, head_starts_at_zero=False, color='k')
    plt.gca().add_patch(ellipse)
    plt.gca().add_patch(arrow2)
    
def draw_observed(state=0, state_label = 0):
    radius = 0.5
    x = state*delta
    y = -2
    pos_circle = plt.Circle((x, y), radius=radius, fill=False, color='g', lw=2)
    plt.gca().add_patch(pos_circle)
    plt.text(x, y, '$X_{%s}$'%(str(state_label)), fontsize=12, 
             horizontalalignment='center', verticalalignment='center')
    arrow = plt.Arrow(x, -radius, 0, -2+2*radius, width=0.25)
    plt.gca().add_patch(arrow)

def draw_infinite(state1=0, state2=1, text=0.1):
    radius = 0.5
    x1 = state1*delta + radius
    diff_state = abs(state2-state1)
    x2 = delta*diff_state - 2*radius
    y = 0
    ellipse = Arc([x1+x2/2,y],x2,0,angle = 0,theta1=0, theta2=180.0,color='k', linewidth='1',ls='dotted')
    arrow2 = FancyArrow(x1+x2/2-0.01, 0, 0.01, 0, width=arrow_width, length_includes_head=False, head_width=None, head_length=None, shape='full', overhang=0, head_starts_at_zero=False, color='k')
    plt.gca().add_patch(ellipse)
    plt.gca().add_patch(arrow2)
    
    y = 0
    ellipse = Arc([1,y],x2,0,angle = 0,theta1=0, theta2=180.0,color='k', linewidth='1',ls='dotted')
    plt.gca().add_patch(ellipse)
    arrow2 = FancyArrow(1-0.01, 0, 0.01, 0, width=arrow_width, length_includes_head=False, head_width=None, head_length=None, shape='full', overhang=0, head_starts_at_zero=False, color='k')
    plt.gca().add_patch(arrow2)

def plot_basic_hmm_model(N = 21, N_states_visible = 10,stay_step_prob = 0.2, one_step_prob = 0.50, two_step_prob = 0.3, circular = False, figsize = (20,10)):
    plt.figure(figsize = figsize)
    state = 1
    
    if circular:
        N_states_visible = N_states_visible - 1
        draw_state(state = state, state_label = N)
        if stay_step_prob>0:
            draw_self_transition(state = state, text = stay_step_prob)
        if one_step_prob>0:
            draw_transition(state1 = state, state2 = state+1, text = one_step_prob)
        if two_step_prob>0:
            draw_transition(state1 = state, state2 = state+2, text = two_step_prob)
        draw_observed(state = state, state_label = N)
    for s in range(N_states_visible):
        state = state + 1
        draw_state(state = state, state_label = state-1)
        if stay_step_prob>0:
            draw_self_transition(state = state, text = stay_step_prob)
        if s<(N_states_visible-1):
            if one_step_prob>0:
                draw_transition(state1 = state, state2 = state+1, text = one_step_prob)
        if s<(N_states_visible-2):
            if two_step_prob>0:
                draw_transition(state1 = state, state2 = state+2, text = two_step_prob)
        draw_observed(state = state, state_label = state-1)

    if circular:
        draw_infinite(state, state+1)
    plt.axis('scaled')
    plt.axis('off')

    plt.show()