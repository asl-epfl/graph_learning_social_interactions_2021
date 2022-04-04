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

error10 = np.load('figs/paper/error.npy')
error10_true = np.load('figs/paper/error_true.npy')
error10_perturbe = np.load('figs/paper/error_perturbe.npy')
error10_perturbe_repeat = np.load('figs/paper/error_perturbe_repeat.npy')
perturbe_time = 7000
perturbe = 2000
eps = 1

times = error10.shape[0]

fig, ax = plt.subplots(3, 1, figsize=(5,10))

ax[0].plot(error10_true, color=GREEN, label='known true state', linewidth=2)
ax[0].plot(error10, color=BLUE, label='learned true state', linewidth=2)
ax[0].set_xlabel('Time')
ax[0].set_ylabel('$\|\|\widetilde{A}_i\|\|^2_{F}$')
#ax[0].legend()
ax[0].grid()

axins = ax[0].inset_axes([0.63, 0.5, 0.3, 0.25])
x1, x2, y1, y2 = 16000, 17000, 0.0744, 0.0884 #0.00012, 0.00019
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.plot(error10_true, color=GREEN, label='case 1: true state', linewidth=2)
axins.plot(error10, color=BLUE, label='case 1: learned true state', linewidth=2)

#ax[1].plot(error10_perturbe, color=BLUE, label='learned true state', linewidth=2)
ax[1].plot(error10_perturbe, color=BLUE, linewidth=2)
ax[1].vlines(perturbe_time, ymin=0 - eps/2, ymax=np.array(error10_perturbe).max() + eps,
             color=ORANGE, label='time when graph changes',
             linestyles='dotted', linewidth=2)
ax[1].set_xlabel('Time')
ax[1].set_ylabel('$\|\|\widetilde{A}_i\|\|^2_{F}$')
#ax[1].legend()
ax[1].grid()

#ax[2].plot(error10_perturbe_repeat, color=BLUE, label='learned true state', linewidth=2)
ax[2].plot(error10_perturbe_repeat, color=BLUE, linewidth=2)
perturbe_list = []
t = perturbe
while t < times:
    perturbe_list.append(t)
    t += perturbe
plt.vlines(perturbe_list, ymin=0 - eps/2, ymax=np.array(error10_perturbe_repeat).max() + eps,
           color=ORANGE,
           linestyles='dotted', linewidth=2)
ax[2].set_xlabel('Time')
ax[2].set_ylabel('$\|\|\widetilde{A}_i\|\|^2_{F}$')
#ax[2].legend()
ax[2].grid()

lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
fig.legend(lines, labels)# loc = (0.15, 0.)

plt.show()
#plt.savefig('figs/paper/errors_all.png', dpi=300, bbox_inches='tight')
