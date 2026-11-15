import sys,pickle,numpy as np,cv2,onnxruntime as ort
from insightface.model_zoo import get_model
from insightface.app.common import Face
S,V,OUT=sys.argv[1],sys.argv[2],sys.argv[3]
start=int(sys.argv[4]) if len(sys.argv)>4 else 0; end=int(sys.argv[5]) if len(sys.argv)>5 else 10**9
d=pickle.load(open(S+'/work/dets.pkl','rb')); members=pickle.load(open(S+'/work/clusters.pkl','rb')); refs=pickle.load(open(S+'/work/refs.pkl','rb'))
cmap={0:'rhino',6:'rhino',1:'negra',2:'negra',3:'negra',5:'negra',7:'negra',4:'psg'}
lab={}
for k,m in enumerate(members):
    for fi,j in m:
        if k in cmap: lab[(fi,j)]=cmap[k]
# temporal kps smoothing per identity
tracks={}
for (fi,j),n in lab.items(): tracks.setdefault(n,{})[fi]=j
skps={}
for n,t in tracks.items():
    for fi,j in t.items():
        ks=[];ws=[]
        for o,w in [(-2,.5),(-1,1),(0,2),(1,1),(2,.5)]:
            if fi+o in t:
                k=d[fi+o][t[fi+o]]['kps']
                if np.abs(k-d[fi][j]['kps']).max()<0.05*(d[fi][j]['bbox'][2]-d[fi][j]['bbox'][0])+3: ks.append(k);ws.append(w)
        skps[(fi,j)]=np.average(ks,axis=0,weights=ws)
swapper=get_model(S+'/models/inswapper_128.onnx',providers=['CPUExecutionProvider'])
src={}
for n,r in refs.items():
    f=Face(kps=r['kps'],bbox=r['bbox']); f.embedding=r['emb']*20; src[n]=f
gf=ort.InferenceSession(S+'/models/gfpgan_1.4.onnx',providers=['CPUExecutionProvider']); gin=gf.get_inputs()[0].name
T=np.array([[0.37691676,0.46864664],[0.62285697,0.46912813],[0.50123859,0.61331904],[0.39308822,0.72541100],[0.61150205,0.72490465]])*512
m=np.zeros((512,512),np.float32); cv2.ellipse(m,(256,280),(170,215),0,0,360,1,-1); MASK=cv2.GaussianBlur(m,(0,0),25)
def enhance(img,kps,alpha):
    M=cv2.estimateAffinePartial2D(kps,T,method=cv2.LMEDS)[0]
    c=cv2.warpAffine(img,M,(512,512),borderMode=cv2.BORDER_REPLICATE)
    x=(c[:,:,::-1].astype(np.float32)/255-.5)/.5
    y=gf.run(None,{gin:x.transpose(2,0,1)[None]})[0][0].transpose(1,2,0)
    y=((np.clip(y,-1,1)+1)/2*255)[:,:,::-1]
    y=c*(1-alpha)+y*alpha
    IM=cv2.invertAffineTransform(M); h,w=img.shape[:2]
    py=cv2.warpAffine(y.astype(np.float32),IM,(w,h)); pm=cv2.warpAffine(MASK,IM,(w,h))[...,None]
    return (img*(1-pm)+py*pm).clip(0,255).astype(np.uint8)
cap=cv2.VideoCapture(V); fps=cap.get(5); W=int(cap.get(3)); H=int(cap.get(4))
cap.set(cv2.CAP_PROP_POS_FRAMES,start); fi=start
wr=cv2.VideoWriter(OUT,cv2.VideoWriter_fourcc(*'mp4v'),fps,(W,H))
while fi<end:
    ok,img=cap.read()
    if not ok: break
    for j,x in enumerate(d[fi]):
        n=lab.get((fi,j))
        if n is None: continue
        k=skps.get((fi,j),x['kps']).astype(np.float32); size=x['bbox'][2]-x['bbox'][0]
        tf=Face(kps=k,bbox=x['bbox'])
        img=swapper.get(img,tf,src[n],paste_back=True)
        if size>=40: img=enhance(img,k,float(np.clip((size-40)/120,0.35,0.8)))
    wr.write(img); fi+=1
    if fi%50==0: print(fi,flush=True)
wr.release()
