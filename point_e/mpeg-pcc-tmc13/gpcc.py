from gpcc_wrapper import load_ply_data, write_ply_data, gpcc_encode, gpcc_decode
import numpy as np
import sys
import h5py
def load_h5(h5_filename):
    f=h5py.File(h5_filename)
    data=f['data'][:]
    return data
#data:n*3
def sort_coors(data):
    #label=data[:,0]+data[:,1]*10+data[:,2]*100
    #print(np.shape(label))
    #idx=np.argsort(label,axis=0)
    ind=np.lexsort(data.T)
    result=data[ind]
    return result
def voxelize(data,gridnum=100):
    mindata=np.min(data,axis=0)
    data=(data-mindata)/(np.max(data,axis=0)-mindata)
    data=data*(gridnum-1)
    result=np.round(data)
    return result
def devoxelize(data,gridnum=100,downbound=-1.0,upbound=1.0):
    data=data/(gridnum-1)
    data=downbound+data*(upbpund-downbound)
    return result
     
if __name__=='__main__':
    prefix='test'
    #data=load_ply_data('/home/xk/codetest/data/plymodels/8iVFB/redandblack_vox10_1550.ply')
    data=load_h5('../../data/hdf5_data/ply_data_test0.h5')[0]
    data=voxelize(data)
    print(np.shape(data))
    #assert False

    #data=sort_coors(data[:10000])
    #data=normal_coors(data)

    y_coords_name = prefix+'_coords.ply'
    write_ply_data(y_coords_name, data)

    y_coords_binname = prefix+'_coords.bin'
    gpcc_encode(y_coords_name,y_coords_binname,False,rate=5,mtype=0)

    gpcc_decode(y_coords_binname,'rec_coords.ply',False,mtype=0)
    #write_ply_data('rec_coords1.ply',sort_coors(load_ply_data('rec_coords.ply')))

