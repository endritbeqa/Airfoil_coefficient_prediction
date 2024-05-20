import os
import shutil


def func():

    source_dir = '/local/disk1/ebeqa/naca_transformer/BluffFOAM-Raw'
    dest_dir = '/local/disk1/ebeqa/naca_transformer/BluffFOAM-Raw_small'

    stl_files = os.listdir(os.path.join(source_dir,'stl'))
    stl_files = sorted(stl_files)
    vtu_files = os.listdir(os.path.join(source_dir,'vtu'))
    vtu_files = sorted(vtu_files)

    for i in range(1500):
        name_stl = os.path.split(stl_files[i])[1]
        name_vtu = os.path.split(vtu_files[i])[1]
        shutil.copy2(os.path.join(source_dir,'stl',stl_files[i]),os.path.join(dest_dir,'stl',name_stl))
        shutil.copy2(os.path.join(source_dir,'vtu',vtu_files[i]),os.path.join(dest_dir,'vtu',name_vtu))


if __name__=='__main__':
    func()
