import numpy as np
import cv2

dragging = False
xi, yi = -1, -1


# Calculates Rotation Matrix given euler angles.
def rotation_matrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta[0]), -np.sin(theta[0])],
                    [0, np.sin(theta[0]), np.cos(theta[0])]
                    ])
    R_y = np.array([[np.cos(theta[1]), 0, np.sin(theta[1])],
                    [0, 1, 0],
                    [-np.sin(theta[1]), 0, np.cos(theta[1])]
                    ])
    R_z = np.array([[np.cos(theta[2]), -np.sin(theta[2]), 0],
                    [np.sin(theta[2]), np.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


class Camera:
    def __init__(self, pos, theta, cameraMatrix, distCoeffs):
        # pose
        self.pos = pos                          # wrt world frame
        self.theta = theta                      # Euler angles: roll pitch yaw
        self.rMat = rotation_matrix(theta)

        self.center = np.zeros(3)               # camera rotates around center
        self.r = np.array([-8., 0., 0.])

        # intrinsic camera parameters
        self.cameraMatrix = cameraMatrix
        self.distCoeffs = distCoeffs

    def set_center(self, vector):
        self.center = vector
        self.pos = np.dot(self.rMat, self.r) + self.center

    def rotate(self, theta):
        self.theta += theta
        self.rMat = rotation_matrix(self.theta)
        self.pos = np.dot(self.rMat, self.r) + self.center
        
    def zoom(self, scl):
        self.r *= scl
        self.pos = self.rMat @ self.r + self.center

    # projects 3d points from world frame to 2d camera image
    def project(self, points):
        # points in frame (in front of the camera) given by a boolean array
        in_frame = np.dot(points - self.pos, self.rMat[:, 0]) > 0.01

        # x-axis is used as projection axis
        M = np.dot(self.rMat, np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]]))

        tvec = -np.dot(np.transpose(M), self.pos)
        rvec = cv2.Rodrigues(np.transpose(M))[0]

        projected_points = cv2.projectPoints(points, rvec, tvec, self.cameraMatrix, self.distCoeffs)[0].astype(np.int64)
        return projected_points, in_frame

    def mouse_control(self, event, x, y, flags, params):
        global xi, yi, dragging
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            xi, yi = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if dragging:
                yaw = 2*np.pi * (x - xi) / 1536
                pitch = -np.pi * (y - yi) / 864
                self.rotate([0, pitch, yaw])
                xi, yi = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            dragging = False
#         elif event == cv2.EVENT_MOUSEWHEEL:
#             if flags < 0:
#                 self.r *= 1.05
#                 self.pos = np.dot(self.rMat, self.r) + self.center
#             elif flags > 0:
#                 self.r /= 1.05
#                 self.pos = np.dot(self.rMat, self.r) + self.center


class Mesh:
    def __init__(self, vertices, edges):
        self.vertices = vertices
        self.edges = edges
        self.pos = np.array([0., 0., 0.])
        self.theta = np.array([0., 0., 0.])

    def draw(self, img, cam, color=(100, 100, 100), pt=1, arrow=False):
        pvertices, in_frame = cam.project(self.vertices)
        for edge in self.edges:
            if in_frame[edge[0]] and in_frame[edge[1]]:
                pt1 = tuple(pvertices[edge[0]][0])
                pt2 = tuple(pvertices[edge[1]][0])
                if arrow:
                    cv2.arrowedLine(img, pt1, pt2, color, pt)
                else:
                    cv2.line(img, pt1, pt2, color, pt)

    def translate(self, vector):
        self.pos += vector
        for vertex in self.vertices:
            vertex += vector

    def rotate(self, theta):
        M1 = np.transpose(rotation_matrix(self.theta))
        M2 = rotation_matrix(theta)
        R = np.dot(M2, M1)
        for vertex in self.vertices:
            delta = self.pos + np.dot(R, vertex - self.pos) - vertex
            vertex += delta
        self.theta = theta


class Force:
    def __init__(self, vertex):
        self.vertex = vertex
        self.F = np.zeros(3)

    def draw(self, img, cam, color=(0, 0, 255), pt=1):
        pt1, in_frame1 = cam.project(np.array([self.vertex]))
        pt2, in_frame2 = cam.project(np.array([self.vertex + self.F]))
        pt1 = tuple(pt1[0][0])
        pt2 = tuple(pt2[0][0])
        # check if pt1 or pt2 are in frame
        if in_frame1[0] and in_frame2[0]:
            cv2.arrowedLine(img, pt1, pt2, color, pt)


def create_grid(rows, cols, length):
    rows, cols = rows+1, cols+1     # extra vertex in each direction
    vertices = np.zeros([rows * cols, 3])
    edges = []
    for i in range(rows):
        for j in range(cols):
            vertices[i * cols + j] = [
                i * length - (rows - 1) * length / 2,
                j * length - (cols - 1) * length / 2,
                0.
            ]
            if i != 0:
                edges.append((cols * (i - 1) + j, cols * i + j))
            if j != 0:
                edges.append((cols * i + j - 1, cols * i + j))
    return Mesh(vertices, np.array(edges))


def create_path(vertices, loop=False):
    edges = [(i, i+1) for i in range(len(vertices)-1)]
    if loop:
        edges.append((0, len(vertices)-1))
    return Mesh(np.array(vertices), np.array(edges))


def create_circle(r, px, py, pz, num=20):
    vertices = np.array([[
        px + r * np.cos(i * 2 * np.pi / num),
        py + r * np.sin(i * 2 * np.pi / num),
        pz
    ] for i in range(num)])
    return create_path(vertices, loop=True)


def group(mesh_list):
    vertices = np.concatenate([
        mesh.vertices for mesh in mesh_list
    ])
    index_shifts = np.cumsum(
        [0] + [len(mesh.vertices) for mesh in mesh_list][:-1]
    )
    edges = np.concatenate([
        mesh.edges + shift for (mesh, shift) in zip(mesh_list, index_shifts)
    ])
    return Mesh(vertices, edges)


def create_drone(r):
    c1 = create_circle(2*r/3, r, -r, 0.)
    c2 = create_circle(2*r/3, -r, -r, 0.)
    c3 = create_circle(2*r/3, r, r, 0.)
    c4 = create_circle(2*r/3, -r, r, 0.)

    l1 = create_path(np.array([[ 2*r/4,  r/3, r/10], [ r, r, 0.]]))
    l2 = create_path(np.array([[ 2*r/4, -r/3, r/10], [ r,-r, 0.]]))
    l3 = create_path(np.array([[-2*r/4, -r/3, r/10], [-r,-r, 0.]]))
    l4 = create_path(np.array([[-2*r/4,  r/3, r/10], [-r, r, 0.]]))
    
    box = create_path(np.array([
        [ 2*r/4,  r/3, r/10],
        [ 2*r/4, -r/3, r/10],
        [-2*r/4, -r/3, r/10],
        [-2*r/4,  r/3, r/10]
    ]), loop=True)
    
    l5 = create_path(np.array([
        [ 2*r/4,          r/3, r/10],
        [ 2*r/4+r/3,  0.7*r/3, r/10],
        [ 2*r/4+r/3, -0.7*r/3, r/10],
        [ 2*r/4,         -r/3, r/10]
    ]))
    
    drone = group([c1, c2, c3, c4, l1, l2, l3, l4, l5, box])
    drone.vertices = np.concatenate([
        drone.vertices,
        np.array([[r, -r, 0.], [r, r, 0.], [-r, r, 0.], [-r, -r, 0.]])  # centers of the circles
    ])

    T1, T2, T3, T4 = drone.vertices[-4:]    # thrust on 4 positions
    #Fg = Force(drone.pos)                   # gravity acts on center of mass
    forces = Force(T1), Force(T2), Force(T3), Force(T4)  #, Fg
    return drone, forces


def set_thrust(drone, forces, T):
    T1, T2, T3, T4 = forces
    T1.F = - T[0] * rotation_matrix(drone.theta)[:, 2]
    T2.F = - T[1] * rotation_matrix(drone.theta)[:, 2]
    T3.F = - T[2] * rotation_matrix(drone.theta)[:, 2]
    T4.F = - T[3] * rotation_matrix(drone.theta)[:, 2]

