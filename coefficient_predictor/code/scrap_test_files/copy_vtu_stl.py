import os
import shutil


def copy_vtu_stl():
    source_dir = '/local/disk1/BluffFOAM/BluffFOAM'
    dest_dir = '/local/disk1/ebeqa/naca_transformer/BluffFOAM-Raw'

    os.makedirs(dest_dir, exist_ok=True)

    vtu_dir = os.path.join(dest_dir,'vtu')
    stl_dir = os.path.join(dest_dir, 'stl')

    os.makedirs(vtu_dir, exist_ok=True)
    #os.makedirs(stl_dir, exist_ok=True)

    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.endswith('.vtu'):
                parent_dir = os.path.split(root)[0]
                parent_dir = os.path.split(parent_dir)[1]
                os.makedirs(os.path.join(vtu_dir,parent_dir), exist_ok=True)
                source_file_path = os.path.join(root,file)
                dest_file_path = os.path.join(vtu_dir,parent_dir, file)
                shutil.copy2(source_file_path,dest_file_path)
            elif file.endswith('.stl'):
                parent_dir = os.path.split(root)[0]
                parent_dir = os.path.split(parent_dir)[0]
                parent_dir = os.path.split(parent_dir)[1]
                os.makedirs(os.path.join(stl_dir,parent_dir), exist_ok=True)
                source_file_path = os.path.join(root,file)
                dest_file_path = os.path.join(stl_dir,parent_dir, file)
                shutil.copy2(source_file_path,dest_file_path)


def process_name():

    target_dir = '/local/disk1/ebeqa/naca_transformer/BluffFOAM-Raw/vtu'

    for root , dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.vtu'):
                parent_dir = os.path.split(root)[1]
                src_path = os.path.join(root, file)
                dst_path = os.path.join(root,parent_dir+'.vtu')
                os.rename(src_path, dst_path)

    target_dir = '/local/disk1/ebeqa/naca_transformer/BluffFOAM-Raw/stl'

    for root , dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.stl'):
                parent_dir = os.path.split(root)[1]
                src_path = os.path.join(root, file)
                dst_path = os.path.join(root,parent_dir+'.stl')
                os.rename(src_path, dst_path)
            
def remove_dir():

    target_dir = '/local/disk1/ebeqa/naca_transformer/BluffFOAM-Raw/vtu'

    for root , dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.vtu'):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(os.path.split(root)[0],file)
                shutil.copy2(src_path, dst_path)

    items = os.listdir(target_dir)

    for item in items:
         item_path = os.path.join(target_dir,item)
         if os.path.isdir(item_path):
             shutil.rmtree(item_path)

    target_dir = '/local/disk1/ebeqa/naca_transformer/BluffFOAM-Raw/stl'

    for root , dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.stl'):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(os.path.split(root)[0],file)
                shutil.copy2(src_path, dst_path)

    items = os.listdir(target_dir)

    for item in items:
         item_path = os.path.join(target_dir,item)
         if os.path.isdir(item_path):
             shutil.rmtree(item_path)


def remove_extra_stl_files():

    stl_dir = '/local/disk1/ebeqa/naca_transformer/BluffFOAM-Raw/stl'
    vtu_dir = '/local/disk1/ebeqa/naca_transformer/BluffFOAM-Raw/vtu'

    stl_items = os.listdir(stl_dir)
    vtu_items = os.listdir(vtu_dir)

    stl_set = set()
    vtu_set = set()

    for item in stl_items:
        stl_set.add(item[:-4])

    for item in vtu_items:
        vtu_set.add(item[:-4])

    extra_stl = list(stl_set.difference(vtu_set))

    for item in extra_stl:
        os.remove(os.path.join(stl_dir, item+'.stl'))


def check_if_same():
    
    stl_dir = '/local/disk1/ebeqa/naca_transformer/BluffFOAM-Raw/stl'
    vtu_dir = '/local/disk1/ebeqa/naca_transformer/BluffFOAM-Raw/vtu'

    stl_items = os.listdir(stl_dir)
    vtu_items = os.listdir(vtu_dir)
    
    stl_set = set() 
    vtu_set = set()

    for item in stl_items:
        stl_set.add(item[:-4])
        

    for item in vtu_items:
        vtu_set.add(item[:-4])

    difference = stl_set.difference(vtu_set)

    print(len(difference))
    #for item in difference:
    #    print(item)

    if stl_set == vtu_set:
        print('The two folders have the same elements')
    else:
        print('The two folders don\'t have the same elements')


if __name__ == '__main__':

    copy_vtu_stl()
    process_name()
    remove_dir()
    remove_extra_stl_files()
    check_if_same()