import scipy.io as sio, numpy as np, pandas as pd, math
M=sio.loadmat('数据/镇龙乡地理空间数据/镇龙乡及周边地理数据/数字高程模型数据（DEM）/镇龙乡及周边30米DEM.mat')
DEM=M['dem'].astype(float); LAT=M['latitude'].ravel(); LON=M['longitude'].ravel(); NOD=float(M['nodata'][0,0])
DLAT=abs(LAT[1]-LAT[0]); DLON=abs(LON[1]-LON[0]); DEM[DEM==NOD]=np.nan
def elev(la,lo):
    i=int(round((LAT[0]-la)/DLAT)); j=int(round((lo-LON[0])/DLON))
    i=min(max(i,0),DEM.shape[0]-1); j=min(max(j,0),DEM.shape[1]-1)
    return float(DEM[i,j])
def profile(la1,lo1,la2,lo2,n=400):
    t=np.linspace(0,1,n); las=la1+(la2-la1)*t; los=lo1+(lo2-lo1)*t
    ii=np.clip(np.round((LAT[0]-las)/DLAT).astype(int),0,DEM.shape[0]-1)
    jj=np.clip(np.round((los-LON[0])/DLON).astype(int),0,DEM.shape[1]-1)
    return DEM[ii,jj]
def haversine(la1,lo1,la2,lo2):
    R=6371000.0
    p1,p2=math.radians(la1),math.radians(la2)
    dp=p2-p1; dl=math.radians(lo2-lo1)
    a=math.sin(dp/2)**2+math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.asin(math.sqrt(a))
