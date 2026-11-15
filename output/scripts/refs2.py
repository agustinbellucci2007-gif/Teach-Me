import sys,pickle,cv2
from insightface.app import FaceAnalysis
S,U=sys.argv[1],sys.argv[2]
app=FaceAnalysis(name='buffalo_l', root=S, allowed_modules=['detection','recognition'], providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1,det_size=(640,640))
out={}
for k,fn in [('main',U+'/eca5f751-image.png'),('driver',U+'/e00e0481-image.png'),('back',U+'/8600ff18-image.png')]:
    fs=sorted(app.get(cv2.imread(fn)),key=lambda x:-(x.bbox[2]-x.bbox[0])); f=fs[0]
    print(k,len(fs),f.bbox.astype(int),round(float(f.det_score),2))
    out[k]=dict(emb=f.normed_embedding,kps=f.kps,bbox=f.bbox)
pickle.dump(out,open(S+'/work/refs2.pkl','wb'))
