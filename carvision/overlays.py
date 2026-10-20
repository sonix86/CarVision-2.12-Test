"""Compact numeric markers with deterministic collision avoidance.

Names belong in the result list/legend, so dense street views stay legible.
"""
def marker_positions(boxes, width, height, size=23):
    placed=[];result=[]
    def overlap(a,b):return a[0]<b[2]+3 and a[2]+3>b[0] and a[1]<b[3]+3 and a[3]+3>b[1]
    for box in boxes:
        x,y,x2,y2=box
        candidates=[(x,y-size-3),(x,y+3),(x2-size,y+3),(x,y2-size-3)]
        candidates += [(x+dx,y+dy) for dy in range(-size*3,size*4,size+5) for dx in range(-size*3,size*4,size+5)]
        candidates += [(a,b) for b in range(0,max(1,height-size+1),size+5) for a in range(0,max(1,width-size+1),size+5)]
        chosen=None
        for a,b in candidates:
            a=max(0,min(max(0,width-size),a));b=max(0,min(max(0,height-size),b));rect=(a,b,a+size,b+size)
            if not any(overlap(rect,p) for p in placed):chosen=rect;break
        # Very small canvases may physically have no space; caller can scroll/zoom.
        chosen=chosen or rect;placed.append(chosen);result.append(chosen)
    return result
