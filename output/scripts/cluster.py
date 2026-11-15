import sys,pickle,numpy as np,cv2
from insightface.app import FaceAnalysis
S=sys.argv[1];U=sys.argv[2];V=sys.argv[3]
d=pickle.load(open(S+'/work/dets.pkl','rb'))
app=FaceAnalysis(name='buffalo_l', root=S, allowed_modules=['detection','recognition'], providers=['CPUExecutionProvider']); app.prepare(ctx_id=-1,det_size=(640,640))
refs={}
for k,fn in [('negra',U+'/168f07e8-image.jpg'),('rhino',U+'/46d91e2f-image.png'),('psg',U+'/61ce377d-image.png')]:
    im=cv2.imread(fn); fs=sorted(app.get(im),key=lambda x:-(x.bbox[2]-x.bbox[0]))
    refs[k]=fs[0]; print(k,len(fs),fs[0].bbox.astype(int))
pickle.dump({k:dict(emb=v.normed_embedding,kps=v.kps,bbox=v.bbox) for k,v in refs.items()},open(S+'/work/refs.pkl','wb'))
# greedy clustering
cents=[];members=[]
for fi,fs in enumerate(d):
    for j,x in enumerate(fs):
        if x['score']<0.5: continue
        sims=[c@x['emb']/np.linalg.norm(c) for c in cents]
        if sims and max(sims)>0.35:
            k=int(np.argmax(sims)); cents[k]=cents[k]+x['emb']; members[k].append((fi,j))
        else: cents.append(x['emb'].copy()); members.append([(fi,j)])
cap=cv2.VideoCapture(V); frames={}
for k,(c,m) in enumerate(zip(cents,members)):
    c=c/np.linalg.norm(c)
    print(k,len(m),'frames',m[0][0],m[-1][0],{r:round(float(c@refs[r].normed_embedding),3) for r in refs})
    if len(m)>=10:
        fi,j=m[len(m)//2]; cap.set(cv2.CAP_PROP_POS_FRAMES,fi); _,f=cap.read()
        b=d[fi][j]['bbox'].astype(int); pad=20
        cv2.imwrite(f"{S}/work/cl{k}.jpg",cv2.resize(f[max(0,b[1]-pad):b[3]+pad,max(0,b[0]-pad):b[2]+pad],(160,160)))
pickle.dump(members,open(S+'/work/clusters.pkl','wb'))
