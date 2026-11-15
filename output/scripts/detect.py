import sys, cv2, numpy as np, pickle
from insightface.app import FaceAnalysis
S=sys.argv[1]; V=sys.argv[2]
app=FaceAnalysis(name='buffalo_l', root=S, allowed_modules=['detection','recognition'], providers=['CPUExecutionProvider'])
app.prepare(ctx_id=-1, det_size=(640,640), det_thresh=0.4)
cap=cv2.VideoCapture(V); out=[]; i=0
while True:
    ok,f=cap.read()
    if not ok: break
    fs=app.get(f)
    out.append([dict(bbox=x.bbox,kps=x.kps,score=x.det_score,emb=x.normed_embedding) for x in fs])
    i+=1
    if i%100==0: print(i,flush=True)
pickle.dump(out,open(S+'/work/dets.pkl','wb'))
print('frames',i)
