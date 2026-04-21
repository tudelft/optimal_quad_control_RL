import numpy as np
import cv2
from . import graphics
import time
import os

# cam matrix
width = int(820*1.5)
height = int(616*1.5)
f=291/1.4*1.5

# graphics
cam = graphics.Camera(
    pos=np.array([-5., 0., 0.]),
    theta=np.zeros(3),
    cameraMatrix=np.array([[f, 0., width/2], [0., f, height/2], [0., 0., 1.]]),
    distCoeffs=np.array([0., 0., 0., 0., 0.])
)
cam.r[0] = -12.

cam.rotate([0., -np.pi/2, 0.])


# grid = graphics.create_grid(10, 10, 0.1)
big_grid = graphics.create_grid(6, 22, 1)

drone, forces = graphics.create_drone(0.08)

# draw axis def
x_axis = graphics.create_path(np.array([[0.,0.,0.],[1.,0.,0.]]))
y_axis = graphics.create_path(np.array([[0.,0.,0.],[0.,1.,0.]]))
z_axis = graphics.create_path(np.array([[0.,0.,0.],[0.,0.,1.]]))

# nxn (m) gate
n = 1
gate = graphics.create_path(np.array([
    [0, n/2, n/2],
    [0, n/2, -n/2],
    [0, -n/2, -n/2],
    [0, -n/2, n/2]
]), loop=True)

# interpolate between two points in 3D
def interpolate(p1, p2, num=10):
    return [p1 + (p2-p1)*i/num for i in range(num+1)]

gate = graphics.create_path(np.array(
    interpolate(np.array([0, n/2, n/2]), np.array([0, n/2, -n/2])) +
    interpolate(np.array([0, n/2, -n/2]), np.array([0, -n/2, -n/2])) +
    interpolate(np.array([0, -n/2, -n/2]), np.array([0, -n/2, n/2])) +
    interpolate(np.array([0, -n/2, n/2]), np.array([0, n/2, n/2]))
), loop=True)
    
# gate_direction = graphics.create_path(np.array([[0,0,0],[.1,0,0]]))
# gate = graphics.group([gate, gate_direction])

# gate collision box
m = 1.5
gate_collision_box_inner = graphics.create_path(np.array([
    [0, m/2, m/2],
    [0, m/2, -m/2],
    [0, -m/2, -m/2],
    [0, -m/2, m/2]
]), loop=True)
m = 2.7
gate_collision_box_outer = graphics.create_path(np.array([
    [0, m/2, m/2],
    [0, m/2, -m/2],
    [0, -m/2, -m/2],
    [0, -m/2, m/2]
]), loop=True)
gate_collision_box = graphics.group([gate_collision_box_inner, gate_collision_box_outer])

scl = 0.2
d = 0.8
b = 1

# options
follow=False
auto_play=False
draw_path=False
draw_forces=False
record=False


def get_drone_state_zero():
    return {
        'x': 0,
        'y': 0,
        'z': 0,
        'phi': 0,
        'theta': 0,
        'psi': 0,
        'u1': 0,
        'u2': 0,
        'u3': 0,
        'u4': 0
    }

def view(get_drone_state=get_drone_state_zero,
         fps=100,
         gate_pos=[],
         gate_yaw=[],
         record_steps=0,
         record_file='output.mp4',
         show_window=True,
         hist_len=100,
         cam_angle=0.,
         reset_func=None,
         gate_size=1,
         grid_bounds=[[-5,5],[-5,5]],
         fake_gates=[],
         ):
    follow=False
    record=False
    draw_forces=True
    draw_path=False
    drone_cam = False
    pause = True
    mask_view = False

    print("""
Keyboard controls:
  SPACE  pause / unpause
  q      reset
  f      toggle follow camera
  d      toggle drone-cam view
  p      toggle path trail
  s      toggle force arrows
  r      toggle video recording
  1 / 2  zoom in / out
  ESC    quit
""")

    # grid
    if grid_bounds is None:
        grid_bounds = [[-5, 5], [-5, 5]]

    x_min = grid_bounds[0][0]
    x_max = grid_bounds[0][1]
    y_min = grid_bounds[1][0]
    y_max = grid_bounds[1][1]
    big_grid = graphics.create_grid(x_max-x_min, y_max-y_min, 1)
    # translate grid to center
    center = [(x_max+x_min)/2, (y_max+y_min)/2, 0]
    big_grid.translate(center)
    
    cam.pos = np.array([-10., 0., 0.])
    cam.theta = np.zeros(3)
    cam.r[0] = -15.
    cam.rotate([0., -np.pi/2, 0.])
    cam.set_center(center)
    
    # translate axis to center
    x_axis.translate(center-x_axis.pos)
    y_axis.translate(center-y_axis.pos)
    z_axis.translate(center-z_axis.pos)
    
    # nxn (m) gate
    gates = []
    for n in gate_size:
        gate = graphics.create_path(np.array([
            [0, n/2, n/2],
            [0, n/2, -n/2],
            [0, -n/2, -n/2],
            [0, -n/2, n/2]
        ]), loop=True)
        gates.append(gate)

    # posistion history
    pos_hist = []
    
    # target point for the drone
    target = graphics.create_path(np.array([[0,0,0],[0,0,0.01]]))

    # videowriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # make sure the record_file directory exists if not create it
    if '/' in record_file:
        os.makedirs(os.path.dirname(record_file), exist_ok=True)             
    out = cv2.VideoWriter(record_file, fourcc, fps=fps, frameSize=(width, height))
    
    # start recording if record_steps is greater than 0
    steps = 0
    if record_steps > 0:
        print('recording started')
        record = True

    # window 
    if show_window:
        cv2.namedWindow('animation', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('animation', cam.mouse_control)
    
    # get initial drone state
    state = get_drone_state()
    
    last_time = time.time()

    while True:
        # ugly hack
        # cam.rotate([0., 0, 0.005])
        # keep track of steps
        steps += 1
        if 0 < record_steps < steps:
            print('recording ended')
            out.release()
            print('recording saved in ' + record_file)
            break
        
        # make sure the loop is fps fps
        current_time = time.time()
        elapsed_time = current_time - last_time
        if elapsed_time < 1/fps:
            time.sleep(float(1/fps) - elapsed_time)
        last_time = time.time()

        # get drone state
        if not pause:
            state = get_drone_state()

        pos = np.stack([state['x'], state['y'], state['z']]).T
        ori = np.stack([state['phi'], state['theta'], state['psi']]).T
        u = np.stack([state['u1'], state['u2'], state['u3'], state['u4']]).T
        
        # add to position history
        pos_hist.pop(0) if len(pos_hist) > hist_len else None
        pos_hist.append(pos)

        # update camera
        if follow:
            cam.set_center(drone.pos)
        else:
            cam.set_center(center)
            
        # drone camera
        if drone_cam:
            cam.pos = drone.vertices[-1] # camera point is the last vertex of the drone
            cam.set_rotation([drone.theta[0], drone.theta[1], drone.theta[2]])
            # IT SHOULD BE:
            R_y = np.array([[np.cos(cam_angle), 0, np.sin(cam_angle)],[0, 1, 0],[-np.sin(cam_angle), 0, np.cos(cam_angle)]])
            cam.rMat = np.dot(cam.rMat, R_y)
            
            
        # using screen resolution of width x height
        frame = 255*np.ones((height, width, 3), dtype=np.uint8)
    
        # draw grid
        big_grid.draw(frame, cam, color=(200, 200, 200), pt=1)
        
        # draw axis
        x_axis.draw(frame, cam, color=(255, 0, 0), pt=2)
        y_axis.draw(frame, cam, color=(0, 255, 0), pt=2)
        z_axis.draw(frame, cam, color=(0, 0, 255), pt=2)
        
        # draw target
        if 'traj_x' in state:
            target_pos = np.stack([state['traj_x'], state['traj_y'], state['traj_z']]).T
            for tp in target_pos:
                target.translate(tp-target.pos)
                target.draw(frame, cam, color=(0,255,0), pt=10)

        # draw all drones
        if len(pos.shape) == 1: # single drone
            drone.translate(pos-drone.pos)
            drone.rotate(ori)
            graphics.set_thrust(drone, forces, u*scl)
            
            if not drone_cam:
                # draw drone
                drone.draw(frame, cam, color=(255, 0, 0), pt=2)

                # draw forces
                if draw_forces:
                    for force in forces:
                        force.draw(frame, cam, color=(0, 0, 255), pt=2)
        else: # multiple drones
            for i in range(pos.shape[0]):
                drone.translate(pos[i]-drone.pos)
                drone.rotate(ori[i])
                graphics.set_thrust(drone, forces, u[i]*scl)

                if not drone_cam:
                    # draw drone
                    if 'color' in state and len(state['color']) > i:
                        drone.draw(frame, cam, color=state['color'][i], pt=2)
                    else:
                        drone.draw(frame, cam, color=(255, 0, 0), pt=2)

                    # draw forces
                    if draw_forces:
                        for force in forces:
                            force.draw(frame, cam, color=(0, 0, 255), pt=2)            
        # draw path
        if draw_path:
            # if multiple drones
            if len(pos.shape) > 1:
                for i in range(pos.shape[0]):
                    path = graphics.create_path([p[i] for p in pos_hist])
                    if 'color' in state and len(state['color']) > i:
                        path.draw(frame, cam, color=state['color'][i], pt=1)
                    else:
                        path.draw(frame, cam, color=(255, 0, 0), pt=1)
            else:
                path = graphics.create_path([p for p in pos_hist])
                if 'color' in state and len(state['color']) > i:
                    path.draw(frame, cam, color=state['color'][i], pt=1)
                else:
                    path.draw(frame, cam, color=(255, 0, 0), pt=1)
                        
        # draw gates
        for i in range(len(gates)):
            pos = gate_pos[i]
            yaw = gate_yaw[i]
            gate = gates[i]
            gate.translate(pos-gate.pos)
            gate.rotate([0,0,yaw])
            gate.draw(frame, cam, color=(0,140,255), pt=4)
            if i not in fake_gates:
                # draw collision box
                gate_collision_box.translate(pos-gate_collision_box.pos)
                gate_collision_box.rotate([0,0,yaw])
                gate_collision_box.draw(frame, cam, color=(255,0,0), pt=4)
                
        # draw a tiny diagram of the drones 4 actuators as pie charts showing motor commands:
        # 1. top left: motor 4
        # 2. top right: motor 2
        # 3. bottom left: motor 3
        # 4. bottom right: motor 1
        if 'u1' in state:
            # only first drone
            if len(state['u1']) == 1:
                u = [state['u1'], state['u2'], state['u3'], state['u4']]
            else:
                u = [state['u1'][0], state['u2'][0], state['u3'][0], state['u4'][0]]
            size = 10
            x_center = width - size*3
            y_center = size*3
            # top left
            cv2.ellipse(frame, (int(x_center-size), int(y_center-size)), (size,size), 0, 0, int(360*u[3]), (0,0,255), -1)
            cv2.ellipse(frame, (int(x_center-size), int(y_center-size)), (size,size), 0, 0, 360, (0,0,0), 1)
            # top right
            cv2.ellipse(frame, (int(x_center+size), int(y_center-size)), (size,size), 0, 0, int(360*u[1]), (0,0,255), -1)
            cv2.ellipse(frame, (int(x_center+size), int(y_center-size)), (size,size), 0, 0, 360, (0,0,0), 1)
            # bottom left
            cv2.ellipse(frame, (int(x_center-size), int(y_center+size)), (size,size), 0, 0, int(360*u[2]), (0,0,255), -1)
            cv2.ellipse(frame, (int(x_center-size), int(y_center+size)), (size,size), 0, 0, 360, (0,0,0), 1)
            # bottom right
            cv2.ellipse(frame, (int(x_center+size), int(y_center+size)), (size,size), 0, 0, int(360*u[0]), (0,0,255), -1)
            cv2.ellipse(frame, (int(x_center+size), int(y_center+size)), (size,size), 0, 0, 360, (0,0,0), 1)
            
        
        if 't' in state:
            cv2.putText(frame, "t = " + str(round(state['t'][0], 2)), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
        if 'v' in state:
            cv2.putText(frame, "v = " + str(round(state['v'][0], 2)), (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))
        # print crash percentage
        if 'crashed' in state:
            cv2.putText(frame, "crash = " + str(round(100*state['crashed'], 1)) + '%', (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

        # recording
        if record:
            out.write(frame)
            cv2.putText(frame, '[recording]', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

        # key events
        key = cv2.waitKeyEx(1)

        # break when esc is pressed
        if key == 27:
            # release videowriter
            if record:
                print('recording ended')
                out.release()
                print('recording saved in ' + record_file)
            break
        # follow when f is pressed
        elif key == ord('f'):
            follow = not follow
        # draw forces when s is pressed
        elif key == ord('s'):
            draw_forces = not draw_forces
        # show drone cam image when d is pressed
        elif key == ord('d'):
            if drone_cam:
                cam.pos = np.array([-10., 0., 0.])
                cam.theta = np.zeros(3)
                cam.r[0] = -15.
                cam.rotate([0., -np.pi/2, 0.])
            # set_cam_f(1000)
            # else:
            #     set_cam_f(200)
            drone_cam = not drone_cam
        # zoom in with 1
        elif key == ord('1'):
            cam.zoom(1.05)
        # zoom out with 2
        elif key == ord('2'):
            cam.zoom(1/1.05)
        # record when r is pressed
        elif key == ord('r'):
            if record:
                print('recording ended')
                out.release()
                print('recording saved in ' + record_file)
            else:
                print('recording started')
            record = not record
        # if p is pressed draw path
        elif key == ord('p'):
            draw_path = not draw_path
        # if space is pressed pause
        elif key == 32:
            pause = not pause
        # if q is pressed we call the reset function
        elif key == ord('q'):
            if reset_func:
                reset_func()
                state = get_drone_state()
                pause = True
        
        # show
        if show_window:
            cv2.imshow('animation', frame)
    cv2.destroyAllWindows()
