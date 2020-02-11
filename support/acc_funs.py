import numpy as np
import pandas as pd

# Function to calculate the AUC
def auc(y,score,both=False):
    s1 = score[np.where(y == 1)[0]]
    s0 = score[np.where(y == 0)[0]]
    den = np.sum(y == 1) * np.sum(y == 0)
    num = 0
    for ss in s1:
        num += np.sum(ss > s0)
        num += np.sum(ss == s0) / 2
    auc = num / den
    if both:
        return(auc, den)
    else:
        return(auc)

def auc_ordinal(y,score):
  score = score[~np.isnan(y)]
  y = y[~np.isnan(y)]
  uy = np.sort(np.unique(y))
  tmp = []
  for yy in uy:
      tmp.append(score[y == yy])
  num, den = 0, 0
  for ii in np.arange(1,len(uy)):
      score_other = np.concatenate(tmp[:ii])
      score_ii = tmp[ii]
      score_broad = score_ii.repeat(len(score_other)).reshape(len(score_ii),len(score_other)).T
      num += np.sum(score_broad > score_other.reshape([len(score_other), 1]))
      num += np.sum(score_broad == score_other.reshape([len(score_other), 1]))/2
      den += np.cumprod(list(score_broad.shape))[1]
  return(num / den)

def pairwise_auc(y,score):
    uy = np.unique(y)
    tmp = []
    for i1 in np.arange(0,len(uy)-1):
        for i2 in np.arange(i1+1,len(uy)):
            #print('i1: %i, i2: %i' % (i1, i2))
            l0, l1 = uy[i1], uy[i2]
            si1 = score[((y == l1) | (y == l0))] #,l1
            yi1 = y[((y == l1) | (y == l0))]
            yi1 = np.where(yi1 == l1, 1, 0)
            ia = auc(y=yi1, score=si1,both=True)
            tmp.append(pd.Series({'y1': l1, 'y0': l0,'n':ia[1], 'auc': ia[0]}))
    df = pd.concat(tmp,axis=1).T
    return (df)

def random_crop(img,height,width,crop_size,ss):
    np.random.seed(ss)
    yidx = np.random.choice(np.arange(height-crop_size))
    xidx = np.random.choice(np.arange(width-crop_size))
    cropped = img[yidx:(yidx+crop_size+1),xidx:(xidx+crop_size)+1].copy()
    xmu = int(xidx+crop_size / 2 )
    ymu = int(yidx + crop_size / 2)
    return(xmu, ymu, cropped)