from plyfile import PlyData
import open3d as o3d
import numpy as np
import sys
import h5py
import os
sys.path.append('..')
from gpcc_wrapper import write_ply_data, gpcc_encode, gpcc_decode
import time
def voxelize(data,gridnum=100):
    colors = data[:,3:]
    data = data[:,:3]
    mindata=np.min(data,axis=0)
    maxdata=np.max(data,axis=0)
    #length=np.max(maxdata-mindata,axis=-1,keepdims=True)
    length=(maxdata-mindata).max()
    data=(data-mindata)/length
    data=data*(gridnum-1)
    #data = np.round(data, decimals=-1)
    result=np.concatenate([data, colors], axis=-1)#np.round(data)
    return result,mindata,length
def devoxelize(data,gridnum,mindata,length):
    colors = data[:,3:]
    data = data[:,:3]
    #print(np.shape(data),gridnum,np.shape(mindata),np.shape(length))
    result=data*length/(gridnum-1)+mindata
    result = np.concatenate([result, colors], axis=-1)
    return result
def load_h5(h5_filename):
    f=h5py.File(h5_filename)
    data=f['data'][:]
    return data
def readgt(path):
    pcd = o3d.io.read_point_cloud(path)
    if np.array(pcd.colors).shape[0] == 0:
        colors = 255.0*np.ones_like(np.array(pcd.points))
    else:
        colors = 255.0*np.array(pcd.colors)
    points = np.concatenate([np.array(pcd.points), colors], axis=-1)
    return points
def load_ply_data(filename):
  '''
  load data from ply file.
  '''

  f = open(filename)
  #1.read all points
  points = []
  for line in f:
    #only x,y,z
    wordslist = line.split(' ')
    try:
      x, y, z = float(wordslist[0]),float(wordslist[1]),float(wordslist[2])
    except ValueError:
      continue
    points.append([x,y,z])
  points = np.array(points)
  points = points.astype(np.float32)#np.uint8
  f.close()

  return points
def write_ply_data(filename, points):
  '''
  write data to ply file.
  '''
  if os.path.exists(filename):
      os.system('rm '+filename)
  f = open(filename,'a+')
  #print('data.shape:',data.shape)
  f.writelines(['ply\n','format ascii 1.0\n'])
  f.write('element vertex '+str(points.shape[0])+'\n')
  f.writelines(['property float x\n','property float y\n','property float z\n'])
  f.write('end_header\n')
  for _, point in enumerate(points):
    f.writelines([str(point[0]), ' ', str(point[1]), ' ',str(point[2]), '\n'])
  f.close()

  return
def write_ply_alpha(filename, points):
    if os.path.exists(filename):
        os.system('rm '+filename)
    f = open(filename,'a+')
    #print('data.shape:',data.shape)
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(points.shape[0])+'\n')
    f.writelines(['property float x\n','property float y\n','property float z\n', 'property float nx\n', 'property float ny\n','property float nz\n'])
    #f.writelines(['property float x\n','property float y\n','property float z\n', 'property uchar red\n', 'property uchar green\n', 'property uchar blue\n'])
    f.write('end_header\n')
    for _, point in enumerate(points):
      #point.astype(np.float16)
      f.writelines([str(point[0]), ' ', str(point[1]), ' ',str(point[2]), ' ', str(point[3]), ' ','0', ' ', '0', '\n'])
      #a=point[3]

      #b=int((a+1)/2)
      #a=a-b
          
      #f.writelines([str(point[0]), ' ', str(point[1]), ' ',str(point[2]), ' ', str(np.uint8(a)), ' ', str(np.uint8(b)), ' ', '0', '\n'])
    f.close()

    return
def load_ply_data(filename):
    points = []
    colors = []
    with open(filename, 'r') as file:
        is_header = True
        has_colors = True

        for line in file:
            if is_header:
                if 'property uchar red' in line:
                    has_colors = True
                if line.strip() == 'end_header':
                    is_header = False
            else:
                values = line.split()
                if has_colors and len(values)<=4:
                    has_colors = False
                points.append([float(values[0]), float(values[1]), float(values[2])])
                if has_colors:
                    colors.append([int(values[3]), int(values[4]), int(values[5])])

    points = np.array(points)
    colors = np.array(colors) if has_colors else None

    if has_colors:
        points = np.concatenate([points, colors], axis=-1)

    return points
def read_ply_point_cloud(filepath):
    # Read the PLY file
    plydata = PlyData.read(filepath)

    # Extract the vertex data (this contains the point cloud data)
    vertex_data = plydata['vertex']

    # Extract the x, y, z coordinates
    x = np.array(vertex_data['x'])
    y = np.array(vertex_data['y'])
    z = np.array(vertex_data['z'])

    # Combine x, y, z into a point cloud array
    points = np.vstack((x, y, z)).T
    #print(vertex_data.data.dtype.names)
    #assert False

    # Check if the file has color attributes (red, green, blue)
    if {'red', 'green', 'blue'}.issubset(vertex_data.data.dtype.names):
        #assert False
        # Extract the colors
        r = np.array(vertex_data['red'])
        g = np.array(vertex_data['green'])
        b = np.array(vertex_data['blue'])

        # Combine r, g, b into a color array
        colors = np.vstack((r, g, b)).T
        return np.concatenate([points, colors], axis=-1)
    else:
        return points
def trans_ply(filename):
    data = read_ply_point_cloud(filename)
    #print(np.shape(data))
    os.system('rm -r '+filename)
    #print(np.shape(data))
    write_ply(filename,data)

def trans_alpha(filename):
    plydata = PlyData.read(filename)
    data=plydata.elements[0].data
    os.system('rm -r '+filename)
    #os.wait()
    #print(np.max(data,axis=1))
    write_ply_alpha(filename,data)
    return result
def gremove(data,pm=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)
    if pm is None:
        pm,_=pcd.segment_plane(distance_threshold=0.01,ransac_n=5,num_iterations=2000)
    a,b,c,d=pm
    result=data[a*data[:,0]+b*data[:,1]+c*data[:,2]+d>0]
    #result=data[data[:,2]>0.1*np.max(data[:,2])]
    return result,pm
def mremove(data,num=1):
    result=[]
    lnum=int(np.sqrt(num))
    minxy=np.min(data,axis=0)
    maxxy=np.max(data,axis=0)
    glen=(maxxy-minxy)/lnum
    pcd = o3d.geometry.PointCloud()
    pms=[]
    ids=[]
    for i in range(lnum):
        for j in range(lnum):
            idx=(data[:,0]>=minxy[0]+i*glen[0]) * (data[:,0]<minxy[0]+(i+1)*glen[0])\
                * (data[:,1]>=minxy[1]+j*glen[1]) * (data[:,1]<minxy[1]+(j+1)*glen[1])
            #print(minxy[0]+i*glen[0],minxy[1]+j*glen[1],maxxy)
            idx=np.where(idx==1)[0]
            #print(idx,i,j)
            dataij=data[idx]
            if np.shape(dataij)[0]>5:
                pcd.points = o3d.utility.Vector3dVector(dataij)
                pm,_=pcd.segment_plane(distance_threshold=0.01,ransac_n=5,num_iterations=2000)
                pms.append(pm)
                a,b,c,d=pm
                idxx=a*dataij[:,0]+b*dataij[:,1]+c*dataij[:,2]+d>0
                ids.append(idx[idxx])
                result.append(dataij[idxx])
    result=np.concatenate(result,axis=0)
    ids=np.concatenate(ids,axis=0)
    return result,pms

#write scales and coordinates
def write_binary(probposi,probs):
    poolen=len(probposi)
    posilen=np.log(poolen)/np.log(2)
    nonzeros=np.sum(probposi)
    spaposi=np.nonzero(probposi)[0]
    result=bitarray()
    #print(result)
    #if nonzeros*posilen+16>poolen:
    #print(poolen,probposi,probs)
    result.frombytes(poolen.to_bytes(2,'big'))
    #print(nonzeros)
    #assert False
    result.extend(probposi)
    for i in range(nonzeros):
        #print(probs[spaposi[i]])
        result.frombytes(int(32767*probs[spaposi[i]]).to_bytes(2,'big'))
    return result
def read_binary(f):
    #ftype=bool.from_bytes(f.read(1),'big')
    poolen=int.from_bytes(f.read(2),'big')
    result=np.zeros(poolen)
    probposi=bitarray()
    #print(poolen)
    probposi.fromfile(f,poolen//8)
    spaposi=np.nonzero(probposi.tolist())[0]
    nonzeros=len(spaposi)

    for i in range(nonzeros):
        result[spaposi[i]]=float(int.from_bytes(f.read(2),'big')/32767)
    return result

def scale_encode(path, drcpath, res=1024):
    data=load_ply_data(path)
    data,mindata,length=voxelize(data,gridnum=res)
    write_ply_data('test.ply',data)
    gpcc_encode('test.ply',drcpath,False,rate=rate,mtype=mtype)

def scale_decode(drcpath, outpath, mindata, length, res=1024):
    gpcc_decode(drcpath,outpath,False,mtype=mtype)
    trans_ply(outpath)
    outdata=load_ply_data(outpath)
    write_ply_data(outpath,devoxelize(outdata,res,mindata,length))

def write_ply(filename, points_colors):
    """
    Write point cloud data with colors to a PLY file.

    Parameters:
    - filename: Path to the output PLY file.
    - points: Numpy array of shape (N, 3), where N is the number of points, and each row contains (X, Y, Z) coordinates.
    - colors: Numpy array of shape (N, 3), where each row contains (R, G, B) values in the range [0, 255].
    """
    # Combine points and colors into a single array
    #points_colors = np.hstack([points, colors])
    # Write to the PLY file
    with open(filename, 'w') as file:
        # Write header
        file.write('ply\n')
        file.write('format ascii 1.0\n')
        file.write(f'element vertex {len(points_colors)}\n')
        file.write('property float x\n')
        file.write('property float y\n')
        file.write('property float z\n')

        if points_colors.shape[-1]>=6:
            file.write('property uchar red\n')
            file.write('property uchar green\n')
            file.write('property uchar blue\n')

        file.write('end_header\n')

        # Write point data (with or without colors)
        if points_colors.shape[-1]>=6:
            for p in points_colors:
                file.write(f'{p[0]} {p[1]} {p[2]} {int(p[3])} {int(p[4])} {int(p[5])}\n')
        else:
            for p in points_colors:
                file.write(f'{p[0]} {p[1]} {p[2]}\n')

def gpcc_com(indir,filedir,outdir,rate=5,mtype=0,res=1024, use_color=False):
    names=os.listdir(indir)
    outdir = os.path.join(outdir, str(rate))
    filedir = os.path.join(filedir,str(rate))
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.exists(filedir):
        os.makedirs(filedir)

    for name in names:
        path=os.path.join(indir,name)

        outpath = os.path.join(outdir, name)
        binpath = os.path.join(filedir, name.split('.')[0]+'.bin')
        
        data=readgt(path)#[:,:3]
        if res is not None:
            data,mindata,length=voxelize(data,gridnum=res)
        write_ply('test.ply',data)

        stime=time.time()
        gpcc_encode('test.ply',binpath,False,rate=6-rate,mtype=mtype, use_color=use_color)
        enctime=time.time()
        # print('compression time: ',enctime-stime)
        gpcc_decode(binpath,outpath,False,mtype=mtype)
        etime=time.time()
        # print('compression time: ',etime-enctime)

        trans_ply(outpath)
        outdata=load_ply_data(outpath)
        #print(outdata.shape)
        #assert False
        if res is not None:
            outdata = devoxelize(outdata,res,mindata,length)
        write_ply(outpath, outdata)

if __name__=='__main__':

    for i in range(1, 6):
        gpcc_com('/dataset/htx/compress/input/objects','/dataset/htx/compress/gpcc/bins/objects','/dataset/htx/compress/gpcc/outplys/objects', rate=i, mtype=0 ,res=512, use_color=True)
