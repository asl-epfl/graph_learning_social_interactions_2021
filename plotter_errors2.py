import numpy as np
import matplotlib.pyplot as plt
import matplotlib

#matplotlib.rcParams.update(matplotlib.rcParamsDefault)

font = {'size': 11.5}
matplotlib.rc('font', **font)

LIGHT_BLUE = "#8ECAE6"
BLUE = "#3A7CB8"#"#219EBC"
DARK_BLUE = "#3A7CB8" #"#035177"
YELLOW = "#FFB703"
ORANGE = "#E5AD02" #"#FB8500"
RED = "#F04532" #"#F00314"
GREEN = "#75A43A" #"#14B37D"

error10_true = np.load('figs/paper/error_true.npy')
error10_state_change = np.load('figs/paper/error_state.npy')
error10_perturbe = np.load('figs/paper/error_perturbe.npy')
error10_perturbe_repeat = np.load('figs/paper/error_perturbe_repeat.npy')
state_change_time = 3000
perturbe_time = 7000
perturbe = 1000
eps = 1

times = error10_true.shape[0]

########################################################################################################################

fig, ax = plt.subplots(1, 1, figsize=(7,5))

ax.plot(error10_true, color=GREEN, label='known true state', linewidth=2)
ax.plot(error10_state_change, color=BLUE, label='learned true state', linewidth=2)
ax.set_xlabel('Time')
ax.set_ylabel('$\|\|\widetilde{A}_i\|\|^2_{F}$')
ax.vlines(state_change_time, ymin=0 - eps/2, ymax=np.array(error10_state_change).max() + eps,
             color=ORANGE, label='time when true state changes',
             linestyles='dotted', linewidth=3)
ax.legend()
ax.grid()

axins = ax.inset_axes([0.68, 0.5, 0.25, 0.25]) #[x0, y0, width, height]
x1, x2, y1, y2 = 4000-70, 4000+70, 0.422, 0.438
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.plot(error10_true, color=GREEN, label='known true state', linewidth=1.5)
axins.plot(error10_state_change, color=BLUE, label='learned true state', linewidth=1.5)
axins.vlines(state_change_time, ymin=0 - eps/2, ymax=np.array(error10_state_change).max() + eps,
             color=ORANGE, label='time when true state changes',
             linestyles='dotted', linewidth=3)

axins = ax.inset_axes([0.28, 0.5, 0.25, 0.25]) #[x0, y0, width, height]
x1, x2, y1, y2 = state_change_time-400, state_change_time+400, 0.55, 0.75
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.plot(error10_true, color=GREEN, label='known true state', linewidth=1.5)
axins.plot(error10_state_change, color=BLUE, label='learned true state', linewidth=1.5)
axins.vlines(state_change_time, ymin=0 - eps/2, ymax=np.array(error10_state_change).max() + eps,
             color=ORANGE, label='time when true state changes',
             linestyles='dotted', linewidth=3)
#plt.show()
plt.savefig('figs/paper/errors_state.png', dpi=300, bbox_inches='tight')

########################################################################################################################

fig, ax = plt.subplots(1, 1, figsize=(7,5))

ax.plot(error10_perturbe, color=BLUE, label='learned true state', linewidth=2)
ax.plot(error10_perturbe, color=BLUE, linewidth=2)
ax.vlines(perturbe_time, ymin=0 - eps/2, ymax=np.array(error10_perturbe).max() + eps,
             color=ORANGE, label='time when graph changes',
             linestyles='dotted', linewidth=3)
ax.set_xlabel('Time')
ax.set_ylabel('$\|\|\widetilde{A}_i\|\|^2_{F}$')
ax.legend()
ax.grid()
#plt.show()
plt.savefig('figs/paper/errors_perturbe.png', dpi=300, bbox_inches='tight')

########################################################################################################################

fig, ax = plt.subplots(1, 1, figsize=(7,5))
ax.plot(error10_perturbe_repeat, color=BLUE, linewidth=2, label='learned true state')
perturbe_list = []
t = perturbe
while t < times:
    perturbe_list.append(t)
    t += perturbe
plt.vlines(perturbe_list, ymin=0 - eps/2, ymax=np.array(error10_perturbe_repeat).max() + eps,
           color=ORANGE, label='time when graph changes',
           linestyles='dotted', linewidth=3)
ax.set_xlabel('Time')
ax.set_ylabel('$\|\|\widetilde{A}_i\|\|^2_{F}$')
ax.legend()
ax.grid()
#plt.show()
plt.savefig('figs/paper/errors_perturbe_repeat.png', dpi=300, bbox_inches='tight')