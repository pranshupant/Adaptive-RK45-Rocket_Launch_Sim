import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import math

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)


def animate(i):
    D = np.loadtxt('data.txt', dtype=float, delimiter='\t')
    print("live_plot")
    ax1.clear()

    try:
        ax1.plot(D[:,0], D[:,-1], 'g-')
        ax1.plot(D[-1,0], D[-1,-1], 'ko', label='Height = %.2f km'%D[-1,-1])
        ax1.plot(0, 0, c='k', label='Velocity = %.2f m/s'%D[-1,1])

        ax1.set_xlim([0, 1500])
        ax1.set_ylim([0, 300])

        plt.title(r'$Trajectory$')
        plt.xlabel(r'$Time$')
        plt.ylabel(r'$Altitude$')
        ax1.legend(loc = 'best')
        ax1.grid()

    except Exception as e:
        print("updating")

ani = animation.FuncAnimation(fig, animate, interval=1)
plt.show()