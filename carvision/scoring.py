"""Exact maximum pairing from section 3 of the assignment."""
import numpy as np
from scipy.optimize import linear_sum_assignment

def canonical_brand(brand):
    value=brand.strip().casefold()
    return {'mercedes':'mercedes-benz','vw':'volkswagen','lixiang':'li auto'}.get(value,value)

def photo_score(reference,predictions):
    if not reference: return 0.0
    n=len(reference); extra=max(0,len(predictions)-n)
    if not predictions: return 0.0
    scores=np.array([[5/n*(.6*(canonical_brand(r[0])==canonical_brand(p[0]))+.4*(r[1].strip().casefold()==p[1].strip().casefold())) for p in predictions] for r in reference])
    i,j=linear_sum_assignment(-scores)
    return round(max(0.,float(scores[i,j].sum())-extra)+1e-10,1)
