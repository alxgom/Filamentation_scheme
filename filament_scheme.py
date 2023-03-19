# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 17:15:21 2022

@author: gomel
"""

# from matplotlib.patches import Circle, PathPatch
# from matplotlib.text import TextPath
# from matplotlib.transforms import Affine2D
# import mpl_toolkits.mplot3d.art3d as art3d
import numpy as np
import matplotlib.pyplot as plt
# from scipy.ndimage import uniform_filter1d
# from matplotlib.colors import LogNorm
# import statistics
# from scipy.stats import mode
# from scipy.signal import find_peaks
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
from matplotlib.text import Annotation
from matplotlib.patches import FancyArrowPatch   
    
    
cm = 1/2.54  # centimeters in inches


plt.rcParams.update({'font.size': 13})
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif"
})


class Arrow3D(FancyArrowPatch):

    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._xyz = (x, y, z)
        self._dxdydz = (dx, dy, dz)

    def draw(self, renderer):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)
        
    def do_3d_projection(self, renderer=None):
        x1, y1, z1 = self._xyz
        dx, dy, dz = self._dxdydz
        x2, y2, z2 = (x1 + dx, y1 + dy, z1 + dz)

        xs, ys, zs = proj_transform((x1, x2), (y1, y2), (z1, z2), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs) 
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)
setattr(Axes3D, 'arrow3D', _arrow3D)

xv=np.linspace(-np.pi,np.pi,300)
yv=np.linspace(-np.pi,np.pi,300)
X, Y = np.meshgrid(xv, yv)
pcolor_matrix=np.zeros((100,100))


def gpulse(xv,yv,sigma):
    gp=np.exp(-(xv**2+yv**2)/(2*sigma)**2)/(sigma*np.sqrt(2*np.pi))
    return gp



theta = np.linspace(0, 2 * np.pi, 201)
radius = 0.1
xc = np.linspace(1, 2, 50)
thetas, xs = np.meshgrid(theta, xc)
yc = radius * np.cos(thetas)
zc = radius * np.sin(thetas)
omeg=np.ones_like(xs)

for j,k in enumerate(np.linspace(4,5.05,len(xs))):
    omeg[j]=omeg[j]*k 

class Annotation3D(Annotation):
    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)

def _annotate3D(ax, text, xyz, *args, **kwargs):
    '''Add anotation `text` to an `Axes3d` instance.'''
    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)

setattr(Axes3D, 'annotate3D', _annotate3D)

fig = plt.figure(figsize=(20*cm,14*cm))
ax = fig.add_subplot(projection='3d')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.view_init(13, -82)
ax.get_yaxis().set_ticks([])
ax.get_zaxis().set_ticks([])
ax.set_axis_off()


gp1=gpulse(X,Y,0.8)
gp2=gpulse(X,Y,1)
gp3=gpulse(X,Y,0.6)
np.linspace(0.5/np.e,1,5)
# [0.5/np.e,1/np.e,1.5/np.e,2/np.e]
arr=np.linspace(0.4/np.e,1,10)

cset = ax.contourf(gp1/np.max(gp3),X, Y ,arr , zdir='x', offset=-0.5, vmin = 0.4/np.e, vmax = 1, cmap='plasma',lw=0,alpha=1,zorder=2)
cset = ax.contourf(gp2/np.max(gp3),X, Y , arr, zdir='x', offset=-1, vmin = 0.4/np.e, vmax = 1, cmap='plasma',lw=0,alpha=1,zorder=1)
cset = ax.contourf(gp3/np.max(gp3),X, Y ,arr, zdir='x', offset=0, vmin = 0.4/np.e, vmax = 1, cmap='plasma',lw=0,alpha=1,zorder=3)
#color=cmap(np.int64(255*np.arange(0.1,1,1/255)))
#norm = BoundaryNorm(np.sort(power/1000), cmap.N)
ax.set_xlim([-1,2.5])
ax.set_zlim([-3,3])

# gpulse(X,Y,0.6)[gpulse(X,Y,0.6)>0.1]
# for i in range(len(xv)):
#     for j in range(len(yv)):
#         if -Z[i,j]>-0: Z2[i,j]=-Z[i,j]
# F=0.2*(X**2+Y**2)-1


ax.arrow3D(-1,0,3.2,
            0,5.1,0,
            mutation_scale=10,
            arrowstyle="-|>",color='gray',
            linestyle='solid')

ax.arrow3D(-1,0,3.2,
            0,0,1.8,
            mutation_scale=10,
            arrowstyle="-|>",color='gray',
            linestyle='solid')

# ax.arrow3D(-1,0,3.2,
            # 4,0,0,
            # mutation_scale=10,
            # arrowstyle="-|>",color='gray',
            # linestyle='solid')


ax.annotate3D(r'$x$', (-1, 5., 3.2), xytext=(-4, 3), textcoords='offset points',zorder=0)
ax.annotate3D(r'$y$', (-1, 0, 5.), xytext=(0, 3), textcoords='offset points',zorder=0)
# ax.annotate3D(r'$z$', (3, 0, 3.2), xytext=(0, 3), textcoords='offset points',zorder=0)

ax.plot(np.linspace(-1,1,len(yv)),2*gpulse(0,yv,1)+3.2,-1,'-k',lw=2,zdir='x',zorder=1)
ax.plot(np.linspace(-1,1,len(yv)),2*gpulse(0,yv,np.sqrt(0.75))+3.2,-0.45,'-k',lw=2,zdir='x',zorder=1)
ax.plot(np.linspace(-1,1,len(yv)),2*gpulse(0,yv,np.sqrt(0.5))+3.2,0,'-k',lw=2,zdir='x',zorder=1)
ax.plot(np.linspace(0,2,len(yv)),0*np.linspace(-1,1,len(yv)),0,'-m',lw=2,zdir='z',alpha=0.9,zorder=0)
ax.get_xaxis().set_visible(True)



theta = np.linspace(0, 2 * np.pi, 201)
radius = 0.1
xc = np.linspace(0, 2, 50)
thetas, xs = np.meshgrid(theta, xc)
yc = radius * np.cos(thetas)
zc = radius * np.sin(thetas)
# ax.plot_surface(xs, np.sin(4*np.pi*(xs-1))*yc, np.sin(4*np.pi*(xs-1))*zc, color='red',alpha=0.8)
ax.plot_surface(xs, np.sin(omeg*np.pi*(xs-1))*yc, np.sin(omeg*np.pi*(xs-1))*zc, color='red',alpha=0.8)




theta = np.linspace(0, 2 * np.pi, 201)
radius = 1
xc = np.linspace(2, 2.8, 201)
thetas, xs = np.meshgrid(theta, xc)
yc = radius * np.cos(thetas)
zc = radius * np.sin(thetas)
ax.plot_surface(xs, (xs-2)*yc, (xs-2)*zc, color='gray',alpha=0.2)

ax.plot([-1,0],[0,0],[-3,-3],'-|k',lw=1,zdir='z',alpha=1,zorder=0)
ax.plot(np.linspace(1,2,len(yv)),0*np.linspace(-1,1,len(yv)),0,'-m',lw=2,zdir='z',alpha=0.9,zorder=0)
ax.plot([0,2],[0,0],[-3,-3],'-|k',lw=1,zdir='z',alpha=1,zorder=0)




ax.arrow3D(-1,0,-3,
            4,0,0,
            mutation_scale=20,
            arrowstyle="-|>",color='black',
            linestyle='solid')

ax.annotate3D('Kerr \n effect', (-1, 0, -4.4), xytext=(3, 3), textcoords='offset points')
ax.annotate3D('Filament', (0.4, 0, -4), xytext=(3, 3), textcoords='offset points')
ax.annotate3D('Propagation \n direction', (2.2, 0, -4.2), xytext=(-3, -3), textcoords='offset points')



plt.show()
fig.savefig('filament_scheme2.svg',bbox_inches='tight')# when saving, specify the DPI\n",
fig.savefig('filament_scheme2.pdf',bbox_inches='tight')# when saving, specify the DPI\n",
fig.savefig('filament_scheme2.png',bbox_inches='tight')# when saving, specify the DPI\n",

