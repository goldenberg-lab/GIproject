# load modules
import os
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


dir_img = os.path.join(os.getcwd(),'..','data','annotated')
os.chdir(dir_img)

# Create gray-scale versions
Image.open('transverse.png').convert('RGB').convert('L').save('gray.png')
Image.open('transverse2.png').convert('RGB').convert('L').save('gray2.png')
diff1 = np.array(Image.open('gray2.png')) - np.array(Image.open('gray.png'))
fig1 = np.array(Image.open('transverse.png').convert('RGB'))
fig2 = np.array(Image.open('transverse2.png').convert('RGB'))
fig2b = fig2.copy()*0+255
id1 = np.where(~(diff1==0))[0]
id2 = np.where(~(diff1==0))[1]
for jj in range(3):
    fig2b[id1,id2,jj] = fig2[id1,id2,jj]
fig2c = fig2b.mean(axis=2)
# Look the figure
fig, axs = plt.subplots(1, 5, figsize=(18, 6))
for ax,ff,zz in zip(axs,[fig1,fig2,diff1,fig2b,fig2c],['CMRmap','CMRmap','gray','CMRmap','gray']):
    ax.imshow(ff,cmap=zz)
# Find the locations where there isn't white space
df = pd.DataFrame(np.where(fig2c < 255)).T.rename(columns={0:'x',1:'y'})
print(df.head())

# Find an optimal cluster number
nclst = np.arange(2,10)
holder = []
for nc in nclst:
    clst = AgglomerativeClustering(n_clusters = nc)
    sc = silhouette_score(df.values,clst.fit_predict(df.values))
    holder.append(sc)
dat_score = pd.DataFrame({'nclust':nclst,'score':holder})
#dat_score.plot.scatter('nclust','score')
nc_star = dat_score.nclust[dat_score.score.idxmax()]
print('Operimal cluster: %i' % nc_star)
# Refit
clst = AgglomerativeClustering(n_clusters = nc_star)
lbls = clst.fit_predict(df.values)
# Find centroid
df_clust = pd.concat([df,pd.DataFrame({'lbls':lbls})],axis=1).groupby('lbls').mean().reset_index()
# Use location from fig2b to get RGB
tmp = []
for ii, rr in df_clust.iterrows():
    tmp.append(fig2b[int(rr['x']),int(rr['y']),:])
df_clust.insert(df_clust.shape[1],'RGB',tmp)
df_clust.insert(df_clust.shape[1],'cell',[''.join(x.astype(str)) for x in df_clust.RGB])
df_clust.cell = df_clust.cell.map(dict(zip(df_clust.cell.unique(),np.arange(len(df_clust.cell.unique()))+4)))

#import seaborn as sns
plt.imshow(fig1)
for ii, rr in df_clust.iterrows():
    plt.scatter(x=int(rr['y']),y=int(rr['x']),s=100,marker=rr['cell']) #c=rr['RGB'].reshape([1,3])/255,
