import plotly.graph_objects as go
import numpy as np
import torch


positions = torch.load("positions_bnerf.pt")
positions = positions.numpy()

theta = torch.load("theta_bnerf.pt")
theta = theta.numpy().flatten()

H,_= positions.shape
x = positions[:,0].flatten()
y = positions[:,1].flatten()
z = positions[:,2].flatten()

Xmax = np.max(x)
Xmin = np.min(x)

Ymax = np.max(y)
Ymin = np.min(y)

Zmax = np.max(z)
Zmin = np.min(z)

X, Y, Z = np.mgrid[Xmin:Xmax:128j, Ymin:Ymax:128j, Zmin:Zmax:128j]
values = X*0

binXsize = (Xmax-Xmin)/127
binYsize = (Ymax-Ymin)/127
binZsize = (Zmax-Zmin)/127

Tmax = np.max(theta)
Tmin = np.min(theta)

threshold =0.00001# de loai bo thanh phan co density rat nho


for i in range(0,H):
    binX,binY,binZ= int((x[i]-Xmin)/binXsize), int((y[i]-Ymin)/binYsize), int((z[i]-Zmin)/binZsize)
    values[binX,binY,binZ]+= theta[i]

fig = go.Figure(data=go.Volume(
    x=X.flatten(),
    y=Y.flatten(),
    z=Z.flatten(),
    value=values.flatten(),
    isomin=Tmin+threshold,
    isomax=Tmax,
    opacity=0.8, # needs to be small to see through all surfaces
    surface_count=17, # needs to be a large number for good volume rendering
    ))
fig.show()
