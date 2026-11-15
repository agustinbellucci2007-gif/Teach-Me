import sys,pickle,numpy as np,cv2
S,IN,OUT=sys.argv[1],sys.argv[2],sys.argv[3]
FR=[int(a) for a in sys.argv[4].split(',')] if len(sys.argv)>4 else None
U='/root/.claude/uploads/842f19db-b6f3-5166-9265-709841e44edf'
d=pickle.load(open(S+'/work/dets.pkl','rb')); members=pickle.load(open(S+'/work/clusters.pkl','rb')); ref=pickle.load(open(S+'/work/refs2.pkl','rb'))['main']
main={}; others={}
for k,m in enumerate(members):
    for fi,j in m:
        if k in (0,6): main[fi]=(j,k)
        elif k in (1,2,3,4,5,7): others.setdefault(fi,[]).append(j)
# smoothed kps (window 7, weights gaussian), reject outliers
sk={}
for fi,(j,k) in main.items():
    ks=[];ws=[]; b=d[fi][j]['bbox']; w=b[2]-b[0]
    for o in range(-3,4):
        if fi+o in main and main[fi+o][1]==k:
            kk=d[fi+o][main[fi+o][0]]['kps']
            if np.abs(kk-d[fi][j]['kps']).max()<0.06*w+3: ks.append(kk); ws.append(np.exp(-o*o/4))
    sk[fi]=np.average(ks,axis=0,weights=ws).astype(np.float32)
photo=cv2.imread(U+'/eca5f751-image.png'); PK=ref['kps'].astype(np.float32)
# --- cap layer in photo coords (hand traced polygon, display coords /4 + (500,20))
poly=np.array([(130,560),(125,430),(160,330),(230,220),(300,150),(330,80),(420,35),(520,15),(620,35),(720,90),(790,190),(820,260),(860,380),(880,500),(880,600),(855,600),(830,450),(740,350),(620,290),(480,275),(330,300),(220,380),(170,520),(160,620)],np.float32)/4+[500,20]
capm=np.zeros(photo.shape[:2],np.float32); cv2.fillPoly(capm,[np.round(poly*8).astype(np.int32)],1.0,lineType=cv2.LINE_AA,shift=3)
capm=cv2.GaussianBlur(capm,(0,0),1.0)
# --- glasses layer drawn in photo coords at 4x
SC=4; H,W=photo.shape[:2]
gl=np.zeros((H*SC,W*SC),np.uint8); tint=np.zeros_like(gl)
def P(pts): return (np.array(pts,np.float32)*SC*8).astype(np.int32)
L=[(541,155),(576,152),(613,150),(612,186),(606,199),(578,201),(552,201),(544,193)]
R=[(628,149),(660,147),(691,147),(692,182),(686,196),(660,198),(636,198),(629,188)]
for poly_ in (L,R):
    cv2.fillPoly(tint,[P(poly_)],255,cv2.LINE_AA,3)
    cv2.polylines(gl,[P(poly_)],True,255,int(2.0*SC),cv2.LINE_AA,3)
    cv2.polylines(gl,[P(poly_[:4])],False,255,int(2.8*SC),cv2.LINE_AA,3)  # thicker top rim
cv2.polylines(gl,[P([(613,159),(620,156),(628,157)])],False,255,int(1.8*SC),cv2.LINE_AA,3)
cv2.polylines(gl,[P([(541,158),(533,160)])],False,255,int(2*SC),cv2.LINE_AA,3)
cv2.polylines(gl,[P([(691,151),(706,154)])],False,255,int(2*SC),cv2.LINE_AA,3)
glm=cv2.resize(gl,(W,H),interpolation=cv2.INTER_AREA).astype(np.float32)/255
tim=cv2.resize(tint,(W,H),interpolation=cv2.INTER_AREA).astype(np.float32)/255
pface=photo[100:300,560:690].reshape(-1,3).mean(0)
def tone(img,mask):
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB).astype(np.float32)
    Lc=lab[...,0]; Lc2=Lc*0.74; hi=Lc2>120; Lc2[hi]=120+(Lc2[hi]-120)*0.5   # darker + less shine
    lab2=np.stack([Lc2,lab[...,1]+4,(lab[...,2]-128)*0.85+128+9],-1).clip(0,255)
    t=cv2.cvtColor(lab2.astype(np.uint8),cv2.COLOR_LAB2BGR).astype(np.float32)
    return img*(1-mask[...,None])+t*mask[...,None]
def skinmask(img):
    ycc=cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
    m=((ycc[...,1]>135)&(ycc[...,1]<180)&(ycc[...,2]>80)&(ycc[...,2]<130)&(ycc[...,0]>40)).astype(np.float32)
    m=cv2.morphologyEx(m,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    return cv2.GaussianBlur(m,(0,0),3)
cap=cv2.VideoCapture(IN); fps=cap.get(5); Wv=int(cap.get(3)); Hv=int(cap.get(4))
wr=None if FR else cv2.VideoWriter(OUT,cv2.VideoWriter_fourcc(*'mp4v'),fps,(Wv,Hv))
fi=0
while True:
    ok,img=cap.read()
    if not ok: break
    if FR and fi not in FR: fi+=1; continue
    if fi in main:
        j,k=main[fi]; kp=sk[fi]; b=d[fi][j]['bbox']; fw=b[2]-b[0]; cx=(b[0]+b[2])/2
        img=img.astype(np.float32)
        # skin tone: face/neck ellipse + skin pixels in this person's body area
        m=np.zeros(img.shape[:2],np.float32)
        ec=(int(cx),int((b[1]+b[3])/2+0.15*fw)); cv2.ellipse(m,ec,(int(0.62*fw),int(0.95*fw)),0,0,360,1,-1)
        body=np.zeros_like(m); cv2.rectangle(body,(int(cx-2.2*fw),int(b[1])),(int(cx+2.2*fw),Hv),1,-1)
        for jo in others.get(fi,[]):
            ob=d[fi][jo]['bbox']; ow=ob[2]-ob[0]; cv2.rectangle(body,(int(ob[0]-.6*ow),int(ob[1]-.6*ow)),(int(ob[2]+.6*ow),Hv),0,-1)
        sm=skinmask(img.astype(np.uint8))
        m=np.maximum(cv2.GaussianBlur(m,(0,0),fw*0.08)*np.maximum(sm,0.85), cv2.GaussianBlur(body,(0,0),fw*0.15)*sm)
        img=tone(img.astype(np.uint8),np.clip(m,0,1)*0.95)
        if k==0:  # not in end shot (already wears sunglasses + cap)
            M=cv2.estimateAffinePartial2D(PK,kp,method=cv2.LMEDS)[0]
            MA=cv2.estimateAffine2D(PK,kp,method=cv2.LMEDS)[0]
            MG=0.5*M+0.5*MA
            anc=np.array([617,92.0]); Sc=np.array([[1.13,0,anc[0]*(1-1.13)],[0,1.13,anc[1]*(1-1.13)-6],[0,0,1]])
            MC=(np.vstack([MG,[0,0,1]])@Sc)[:2]
            occl=d[fi][j]['score']<0.75
            scale=np.sqrt(abs(np.linalg.det(M[:,:2])))
            fr=img[int(max(0,b[1])):int(b[3]),int(max(0,b[0])):int(b[2])].reshape(-1,3).mean(0)
            gain=np.clip(fr/pface,0.15,1.2)
            capimg=cv2.warpAffine(photo.astype(np.float32)*gain,MC,(Wv,Hv),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
            ca=cv2.warpAffine(capm,MC,(Wv,Hv))
            sig=max(0.0,0.6/scale-0.3)
            if sig>0.2: capimg=cv2.GaussianBlur(capimg,(0,0),sig); ca=cv2.GaussianBlur(ca,(0,0),sig)
            ga=cv2.warpAffine(glm,MG,(Wv,Hv))*(0 if occl else 1); ta=cv2.warpAffine(tim,MG,(Wv,Hv))*(0 if occl else 1)
            img=img*(1-ta[...,None]*0.10)+np.array([60,60,60])*ta[...,None]*0.10
            # small specular on lens top
            img=img*(1-ga[...,None]*0.95)+np.array([22,22,24])*ga[...,None]*0.95
            img=img*(1-ca[...,None])+capimg*ca[...,None]
        img=img.clip(0,255).astype(np.uint8)
    if FR: cv2.imwrite(f'{OUT}_{fi}.jpg',img)
    else: wr.write(img)
    fi+=1
    if fi%100==0: print(fi,flush=True)
if wr: wr.release()
