from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import h5py
import numpy as np

with h5py.File("optimalFigure8.jld", "r") as f:
    dimT = f["state"].shape[0]
    dimX = f[f["state"][0]].shape[0]
    dimU = f[f["control"][0]].shape[0]
    state = np.zeros((dimT, dimX), dtype=np.float64)
    control = np.zeros((dimT, dimU), dtype=np.float64)
    tf = f["tf"][()]
    objective = f["objective"][()]
    for i, (state_ref, control_ref) in enumerate(zip(f["state"], f["control"])):
        state[i] = np.array(f[state_ref])
        control[i] = np.array(f[control_ref])


fig = plt.figure(figsize=(4.5, 5))
#plt.subplots_adjust(wspace=0.3, bottom=0.2)
ax = fig.add_subplot(111)#, projection='3d')
ax.set_title("Optimal Trajectory")
#axCtl = fig.add_subplot(122)

gates = [
    [(0, 0.75), (0, 0)],
    [(1.5, 1.5), (0.75, 0.75+1.5)],
    [(1.5+0.75, 3.75), (0, 0)],
    [(1.5, 1.5), (-0.75, -0.75-1.5)],
    ]

for gate in gates:
    ax.plot(gate[1], gate[0], c="black")
    ax.plot(gate[1], (-gate[0][0], -gate[0][1]), c="black")

x, y, z = state[:, :3].T
vx, vy, vz = state[:, 3:6].T
v = np.linalg.norm(state[:, 3:6], axis=1)
norm = Normalize(vmin=0., vmax=max(v))
colors = plt.cm.jet(norm(v))
for i, c in enumerate(colors[:-1]):
    #sc = ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], c=c, label="path")
    sc = ax.plot(y[i:i+2],  (x[i:i+2]+1.5), c=c, label="path")
    sc = ax.plot(y[i:i+2], -(x[i:i+2]+1.5), c=c, label="path")

ax.set_xlabel("x (m)")
ax.set_ylabel("y (m)")
ax.axis('equal')
ax.set(xlim=(-2.5, 2.5), ylim=(-4, 4))
ax.grid(True)
#ax.set_zlabel("Z [m]")

# dumb colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=norm)
sm.set_array([])  # Required for ScalarMappable to work

# Add colorbar to the figure
cbar = plt.colorbar(sm, ax=ax)#, shrink=0.6, pad=0.2)
cbar.set_label("Speed (m/s)")

annot = ax.annotate(f"$V_{{mean}}$ = {v.mean() :.2f}\n $V_{{max}}$ = {v.max() :.2f}", (1,3.1), size=9)

#ax.view_init(elev=-167, azim=-121)

# Set equal scaling for all axes
#max_range = np.array([max(x)-min(x), max(y)-min(y), max(z)-min(z)]).max() / 2.0
#mid_x = (max(x) + min(x)) * 0.5
#mid_y = (max(y) + min(y)) * 0.5
#mid_z = (max(z) + min(z)) * 0.5

#ax.set_xlim(mid_x - max_range, mid_x + max_range)
#ax.set_ylim(mid_y - max_range, mid_y + max_range)
#ax.set_zlim(mid_z - max_range, mid_z + max_range)


#axCtl.plot(np.linspace(0, tf, dimT), control)
#axCtl.set_xlabel("Time")
#axCtl.set_ylabel("Control")
#axCtl.legend([f"Motor {i}" for i in range(1,5)])
#axCtl.grid()

fig.savefig("path.pdf", format='pdf')

