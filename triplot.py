from RegretMin import *
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import time

proj = np.array(
[[-1 * np.cos(30. / 360. * 2. * np.pi),np.cos(30. / 360. * 2. * np.pi),0.],
[-1 * np.sin(30. / 360. * 2. * np.pi),-1 * np.sin(30. / 360. * 2. * np.pi),1.]])

def projplot(p, ax, colour="blue", marker="x", triangle = False, boundary=False):
    global proj

    if boundary:
        ts = np.linspace(0, 1, 2000)
        PBd1 = proj @ np.array([ts,(1-ts),0*ts])
        PBd2 = proj @ np.array([0*ts,ts,(1-ts)])
        PBd3 = proj @ np.array([ts,0*ts,(1-ts)])
        ax.plot(PBd1[0], PBd1[1], ".",color='black',markersize=3, alpha=0.01)
        ax.plot(PBd2[0], PBd2[1], ".",color='black',markersize=3, alpha=0.01)
        ax.plot(PBd3[0], PBd3[1], ".",color='black',markersize=3, alpha=0.01)
        ax.text(-1, -0.55, "$x_1$", fontsize=15)
        ax.text(0.9, -0.55, "$x_2$", fontsize=15)
        ax.text(-0.05, 1.05, "$x_3$", fontsize=15)
    elif triangle:
        vert = proj @ p
        poly = Polygon(vert.T, color=colour)
        return poly

    else:
        pproj = proj @ p.T
        ax.scatter(pproj[0], pproj[1], s=60, c=colour, marker=".")


""" fig, ax = plt.subplots()
projplot(None, ax, boundary=True) """

fig, (ax1, ax2) = plt.subplots(1,2)
fig.set_size_inches(20, 6)
ax1.set_xlim(-1.2, 1.2)
ax1.set_ylim(-0.7, 1.2)
ax2.set_xlim(-1.2, 1.2)
ax2.set_ylim(-0.7, 1.2)
projplot(None, ax1, boundary=True)
projplot(None, ax2, boundary=True)
P = payoffmatrix()


def plotpoints(s, ax):
    smap = s - np.min(s)
    smap /= np.max(smap)
    colours = np.vstack((1 - smap, smap, np.zeros(N))).T
    projplot(actions/S, ax, colour=colours)

def stats(s1, s2, r1, r2):
    print("s1Ps2'", s1 @ (P @ s2.T))

    S1, S2 = np.meshgrid(s1,s2)
    pij = S1 * S2
    print(P @ pij.T)

    plt.show()


start = time.time()
s1, s2, r = doubletrain(100000)
s1, s2 = getAverageStrategy(s1), getAverageStrategy(s2)
print(time.time() - start)

plotpoints(s1, ax1)
plotpoints(s2, ax2)


plt.show()

