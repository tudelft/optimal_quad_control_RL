from matplotlib import pyplot as plt
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


fig = plt.figure(figsize=(10, 4))
plt.subplots_adjust(wspace=0.3, bottom=0.2)
ax = fig.add_subplot(121, projection='3d')
axCtl = fig.add_subplot(122)

x, y, z = state[:, :3].T
ax.plot(x, y, z, label="path")
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")

ax.view_init(elev=-167, azim=-121)

# Set equal scaling for all axes
max_range = np.array([max(x)-min(x), max(y)-min(y), max(z)-min(z)]).max() / 2.0
mid_x = (max(x) + min(x)) * 0.5
mid_y = (max(y) + min(y)) * 0.5
mid_z = (max(z) + min(z)) * 0.5

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)


axCtl.plot(np.linspace(0, tf, dimT), control)
axCtl.set_xlabel("Time")
axCtl.set_ylabel("Control")
axCtl.legend([f"Motor {i}" for i in range(1,5)])
axCtl.grid()

fig.savefig("path.pdf", format='pdf')

