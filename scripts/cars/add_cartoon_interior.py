#!/usr/bin/env python3
"""
Adds an interior to the ported "Modern Cartoon Sports Car" (scripts/cars/port_cartoon_sports_car.py output): seats,
dashboard with an instrument hood and two gauges, centre tunnel, door cards, headliner, mirror and a steering wheel as
its own `STEER` node (column along its local +Z, tilted 24 degrees; the game turns it about that axis). Also moves the
cockpit and dash cameras inside the cabin.

  python3 scripts/cars/add_cartoon_interior.py IN.glb OUT.glb
  python3 scripts/cars/remove_cartoon_lid.py OUT.glb FINAL.glb   # then: removes the source's opaque sheet over the cabin
"""
import sys
import json,struct,math
import numpy as np
def load(p):
    f=open(p,'rb'); f.read(12)
    l,t=struct.unpack('<II',f.read(8)); j=json.loads(f.read(l))
    l2,t2=struct.unpack('<II',f.read(8)); b=f.read(l2); return j,b
def acc(j,b,i):
    a=j['accessors'][i]; bv=j['bufferViews'][a['bufferView']]
    n={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}[a['type']]
    dt={5126:'<f4',5123:'<u2',5125:'<u4',5121:'u1'}[a['componentType']]
    off=bv.get('byteOffset',0)+a.get('byteOffset',0)
    st=bv.get('byteStride')
    if st and st!=np.dtype(dt).itemsize*n:
        raw=np.frombuffer(b,dtype='u1',count=a['count']*st,offset=off).reshape(a['count'],st)[:,:np.dtype(dt).itemsize*n]
        return np.frombuffer(raw.tobytes(),dtype=dt).reshape(a['count'],n)
    return np.frombuffer(b,dtype=dt,count=a['count']*n,offset=off).reshape(a['count'],n)
def local(n):
    if 'matrix' in n: return np.array(n['matrix'],float).reshape(4,4).T
    M=np.eye(4)
    t=n.get('translation',[0,0,0]); r=n.get('rotation',[0,0,0,1]); s=n.get('scale',[1,1,1])
    x,y,z,w=r
    R=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    M[:3,:3]=R*np.array(s); M[:3,3]=t; return M

SRC, OUT = sys.argv[1], sys.argv[2]
f=open(SRC,'rb'); f.read(12); l,t=struct.unpack('<II',f.read(8)); J=json.loads(f.read(l)); l2,t2=struct.unpack('<II',f.read(8)); BIN=bytearray(f.read(l2))
def rotx(a): c,s=math.cos(a),math.sin(a); return np.array([[1,0,0],[0,c,-s],[0,s,c]])
def rot_to(z,ax):  # rotation matrix taking +Z to unit vector ax
    z=np.array([0,0,1.]); ax=np.array(ax,float)/np.linalg.norm(ax); v=np.cross(z,ax); c=z@ax
    if np.linalg.norm(v)<1e-9: return np.eye(3) if c>0 else np.diag([1,-1,-1.])
    vx=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]]); return np.eye(3)+vx+vx@vx*(1/(1+c))
class M:
    def __init__(s): s.P=[];s.N=[];s.I=[]
    def add(s,p,n,i):
        o=sum(len(x) for x in s.P); s.P.append(np.asarray(p,float)); s.N.append(np.asarray(n,float)); s.I.append(np.asarray(i,np.uint32)+o)
    def arrays(s):
        P=np.vstack(s.P);N=np.vstack(s.N);I=np.concatenate(s.I).reshape(-1,3).copy()
        # fix winding so faces agree with vertex normals
        e1=P[I[:,1]]-P[I[:,0]];e2=P[I[:,2]]-P[I[:,0]];fn=np.cross(e1,e2);vn=N[I[:,0]]+N[I[:,1]]+N[I[:,2]]
        bad=(fn*vn).sum(1)<0; I[bad]=I[bad][:,[0,2,1]]
        return P.astype('<f4'),N.astype('<f4'),I.reshape(-1).astype('<u4')
    def box(s,c,sz,R=np.eye(3)):
        c=np.array(c,float);h=np.array(sz,float)/2
        for a in range(3):
            for sg in (1,-1):
                u=np.zeros(3);v=np.zeros(3);n=np.zeros(3);n[a]=sg;u[(a+1)%3]=1;v[(a+2)%3]=1
                q=[-u*h[(a+1)%3]-v*h[(a+2)%3],u*h[(a+1)%3]-v*h[(a+2)%3],u*h[(a+1)%3]+v*h[(a+2)%3],-u*h[(a+1)%3]+v*h[(a+2)%3]]
                q=[x+n*h[a] for x in q]; idx=[0,1,2,0,2,3] if sg>0 else [0,2,1,0,3,2]
                s.add([c+R@x for x in q],[R@n]*4,idx)
    def cyl(s,p0,p1,r,seg=18,caps=True):
        p0=np.array(p0,float);p1=np.array(p1,float);ax=p1-p0;R=rot_to(0,ax);L=np.linalg.norm(ax)
        ang=np.linspace(0,2*np.pi,seg,endpoint=False);ring=np.c_[np.cos(ang)*r,np.sin(ang)*r,np.zeros(seg)]
        P=[];N=[];I=[]
        for k in range(seg): 
            P+= [ring[k],ring[k]+[0,0,L]]; nn=np.array([math.cos(ang[k]),math.sin(ang[k]),0]);N+=[nn,nn]
        for k in range(seg):
            a=2*k;b=2*((k+1)%seg);I+=[a,b,a+1,b,b+1,a+1]
        o=len(P)
        if caps:
            for z,nz in ((0,-1),(L,1)):
                c0=len(P);P.append([0,0,z]);N.append([0,0,nz])
                for k in range(seg): P.append(ring[k]+[0,0,z]);N.append([0,0,nz])
                for k in range(seg): I+=[c0,c0+1+k,c0+1+(k+1)%seg]
        P=np.array(P);N=np.array(N);s.add((R@P.T).T+p0,(R@N.T).T,I)
    def torus(s,R_,r_,Rot,c,su=40,sv=12):
        P=[];N=[];I=[]
        for i in range(su):
            u=2*np.pi*i/su
            for j in range(sv):
                v=2*np.pi*j/sv; cx=np.array([math.cos(u),math.sin(u),0])
                P.append(cx*(R_+r_*math.cos(v))+np.array([0,0,r_*math.sin(v)])); N.append(cx*math.cos(v)+np.array([0,0,math.sin(v)]))
        for i in range(su):
            for j in range(sv):
                a=i*sv+j;b=((i+1)%su)*sv+j;c=((i+1)%su)*sv+(j+1)%sv;d=i*sv+(j+1)%sv;I+=[a,b,c,a,c,d]
        P=np.array(P);N=np.array(N);s.add((Rot@P.T).T+c,(Rot@N.T).T,I)
    def disc(s,c,n,r,seg=24):
        n=np.array(n,float);R=rot_to(0,n);P=[[0,0,0]]+[[math.cos(a)*r,math.sin(a)*r,0] for a in np.linspace(0,2*np.pi,seg,endpoint=False)]
        I=[]
        for k in range(seg): I+=[0,1+k,1+(k+1)%seg]
        P=np.array(P);s.add((R@P.T).T+np.array(c,float),[n]*len(P),I)
MATS={}
def mat(name,col,rough=0.8,metal=0.0,emis=None,ds=True):
    m={'name':name,'doubleSided':ds,'pbrMetallicRoughness':{'baseColorFactor':list(col)+[1.0],'metallicFactor':metal,'roughnessFactor':rough}}
    if emis: m['emissiveFactor']=list(emis)
    J['materials'].append(m); MATS[name]=len(J['materials'])-1
mat('cabin_dark',(0.035,0.035,0.04),0.85)
mat('cabin_liner',(0.30,0.30,0.31),0.95)
mat('cabin_seat',(0.045,0.045,0.055),0.9)
mat('cabin_accent',(0.55,0.025,0.025),0.7)
mat('cabin_wheel',(0.02,0.02,0.022),0.55)
mat('cabin_gauge',(0.01,0.01,0.012),0.4,emis=(0.05,0.55,0.75))
mat('cabin_chrome',(0.55,0.55,0.58),0.25,metal=0.9)
mat('cabin_mirror',(0.62,0.66,0.70),0.08,metal=0.95)
G={k:M() for k in MATS}
# ----- layout (rig space: +X left, +Y up, +Z forward); driver on the left
G['cabin_dark'].box((0,0.38,-0.10),(1.90,0.04,2.30))                   # floor
G['cabin_liner'].box((0,1.62,-0.35),(1.90,0.03,1.30))                  # headliner
G['cabin_dark'].box((0,0.72,-1.10),(1.90,0.68,0.04))                   # back panel
G['cabin_dark'].box((0,1.02,-1.00),(1.90,0.06,0.30))                   # parcel shelf
for sx in (1,-1):
    G['cabin_dark'].box((sx*0.96,0.72,-0.15),(0.04,0.66,1.90))         # door cards
    G['cabin_accent'].box((sx*0.935,0.80,-0.15),(0.02,0.06,1.20))      # door trim stripe
G['cabin_dark'].box((0,0.84,0.94),(1.96,0.28,0.56))                    # dashboard body
G['cabin_dark'].box((0,0.995,0.92),(1.90,0.05,0.50),rotx(0.10))        # dash pad (slopes to the glass)
G['cabin_dark'].box((0.38,1.06,0.80),(0.52,0.13,0.20))                 # instrument hood
for gx in (0.27,0.49):
    G['cabin_gauge'].disc((gx,1.06,0.698),(0,0,-1),0.075)              # two gauges facing the driver
G['cabin_accent'].box((0,0.90,0.665),(1.60,0.025,0.01))                # dash trim stripe
G['cabin_dark'].box((0,0.52,0.10),(0.20,0.28,1.50))                    # centre tunnel
G['cabin_chrome'].cyl((0,0.66,0.22),(0,0.80,0.22),0.016)               # gear lever
G['cabin_wheel'].disc((0,0.80,0.22),(0,1,0),0.0)  if False else None
G['cabin_wheel'].box((0,0.83,0.22),(0.05,0.05,0.05))                   # gear knob
for sx in (0.38,-0.38):
    G['cabin_seat'].box((sx,0.46,-0.10),(0.52,0.12,0.60))             # cushion
    tilt=rotx(-0.21)                                                   # backrest leans back
    G['cabin_seat'].box((sx,0.82,-0.40),(0.52,0.62,0.12),tilt)
    G['cabin_accent'].box((sx,0.80,-0.335),(0.30,0.46,0.02),tilt)      # red insert
    G['cabin_seat'].box((sx,1.28,-0.46),(0.30,0.22,0.10),tilt)         # headrest
    G['cabin_dark'].box((sx,0.40,-0.10),(0.14,0.30,0.50))              # seat rail block
# interior mirror: hangs 12 cm under the glass on a short arm, turned and tipped towards the driver
Rm=np.array([[math.cos(-0.18),0,math.sin(-0.18)],[0,1,0],[-math.sin(-0.18),0,math.cos(-0.18)]])@rotx(-0.12)
G['cabin_dark'].box((0,1.31,0.80),(0.27,0.078,0.04),Rm)
G['cabin_mirror'].box((0,1.31,0.80)+Rm@np.array([0,0,-0.0215]),(0.245,0.056,0.004),Rm)
G['cabin_dark'].cyl((0,1.42,0.84),(0,1.34,0.80),0.011,10)
G['cabin_dark'].box((0,1.42,0.855),(0.05,0.03,0.016))
# ----- steering wheel (own node, column along local +Z, tilted 24 deg down towards the dash)
SW={k:M() for k in ('cabin_wheel','cabin_accent','cabin_dark')}
ID=np.eye(3)
SW['cabin_wheel'].torus(0.185,0.019,ID,(0,0,0))
for ang in (90,210,330):
    a=math.radians(ang);SW['cabin_wheel'].cyl((0,0,0),(math.cos(a)*0.175,math.sin(a)*0.175,0),0.014,10,False)
SW['cabin_dark'].cyl((0,0,-0.03),(0,0,0.03),0.055,20)
SW['cabin_dark'].cyl((0,0,0.03),(0,0,0.34),0.028,14)
SW['cabin_accent'].box((0,0.185,0),(0.025,0.045,0.045))
# ----- write
def accessor(arr,typ,target,minmax=False):
    global BIN
    while len(BIN)%4: BIN.append(0)
    off=len(BIN); data=arr.tobytes(); BIN+=data
    J['bufferViews'].append(dict(buffer=0,byteOffset=off,byteLength=len(data),target=target))
    ct={'<f4':5126,'<u4':5125}[arr.dtype.str]; a=dict(bufferView=len(J['bufferViews'])-1,componentType=ct,count=int(arr.shape[0]),type=typ)
    if minmax: a['min']=arr.min(0).astype(float).tolist();a['max']=arr.max(0).astype(float).tolist()
    J['accessors'].append(a); return len(J['accessors'])-1
def prims(groups):
    out=[]
    for k,m in groups.items():
        if not m.P: continue
        P,N,I=m.arrays()
        out.append(dict(attributes=dict(POSITION=accessor(P,'VEC3',34962,True),NORMAL=accessor(N,'VEC3',34962)),indices=accessor(I.reshape(-1,1),'SCALAR',34963),material=MATS[k]))
    return out
body=[n for n in J['nodes'] if n['name']=='BODY'][0]
J['meshes'][body['mesh']]['primitives']+=prims(G)
J['meshes'].append(dict(name='STEER',primitives=prims(SW)))
a=math.radians(24); q=[math.sin(a/2),0,0,math.cos(a/2)]
J['nodes'].append(dict(name='STEER',mesh=len(J['meshes'])-1,translation=[0.38,0.93,0.50],rotation=q))
car=[n for n in J['nodes'] if n['name']=='CAR'][0]; car['children'].append(len(J['nodes'])-1)
cam=car['extras']['cameras']
cam['cockpit']=dict(pos=[0.38,1.22,-0.05],pitchDeg=-3,fov=60)
cam['dash']=dict(pos=[0.38,1.27,0.85],pitchDeg=-2,fov=58)
car['extras']['source']['changes']+='; interior added (seats, dashboard, instrument hood, steering wheel, console, door cards, headliner)'
J['buffers'][0]['byteLength']=len(BIN)
js=json.dumps(J,separators=(',',':')).encode()
while len(js)%4: js+=b' '
while len(BIN)%4: BIN.append(0)
open(OUT,'wb').write(struct.pack('<III',0x46546C67,2,12+8+len(js)+8+len(BIN))+struct.pack('<II',len(js),0x4E4F534A)+js+struct.pack('<II',len(BIN),0x004E4942)+bytes(BIN))
print('ok',len(BIN)//1024,'KB bin; new prims',len(J['meshes'][body['mesh']]['primitives']))
