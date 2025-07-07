import os
import pickle
import SimpleITK as sitk
import pandas as pd
import shutil

def get_image_array(image_path):
    '''
    This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
    '''
    
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(image_path)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    return ct_scan

def save_pkl(x_image, y_image, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump((x_image, y_image), f)

def pkload(fname):
    with open(fname, 'rb') as f:
        return pickle.load(f)

def get_ids(split_path):
    id_list = pd.read_csv(split_path)
    id_list = list(id_list['ID'].values)
    id_list = [i.split(".")[0][:-1] for i in id_list]
    return id_list

def normal2normal(ids, ct_type, moved, fixed, result_folder):
    data_folder = r'/workspace/data/orCaScore/Challenge_data/Training_set/Images'
    for i in ids:
        x_path = os.path.join(data_folder, f"{i}{ct_type[moved]}.mhd")
        x_image = get_image_array(x_path)
        y_path = os.path.join(data_folder, f"{i}{ct_type[fixed]}.mhd")
        y_image = get_image_array(y_path)
        save_pkl(x_image, y_image, os.path.join(result_folder, f"{i}.pkl"))

def normalized2normalized(ids, ct_type, moved, fixed, result_folder):
    data_folder = r'/workspace/data/orCaScore/Challenge_data/Training_set/Images'
    for i in ids:
        x_path = os.path.join(data_folder, f"{i}{ct_type[moved]}.mhd")
        x_image = get_image_array(x_path)
        x_image = normalize_image(x_image)
        y_path = os.path.join(data_folder, f"{i}{ct_type[fixed]}.mhd")
        y_image = get_image_array(y_path)
        y_image = normalize_image(y_image)
        save_pkl(x_image, y_image, os.path.join(result_folder, f"{i}.pkl"))
        
def transformed2normal(ids, ct_type, fixed, result_folder):
    x_folder = r'/workspace/data/cytran/net_A/cytran_pos_orca/images'
    y_folder = r'/workspace/data/orCaScore/Challenge_data/Training_set/Images'
    
    for i in ids:
        x_image = pkload(os.path.join(x_folder, f"{i}.pkl"))
        y_path = os.path.join(y_folder, f"{i}{ct_type[fixed]}.mhd")
        y_image = get_image_array(y_path)
        y_image = normalize_image(y_image)
        save_pkl(x_image, y_image, os.path.join(result_folder, f"{i}.pkl"))
        
def normal2transformed(ids, ct_type, moved, result_folder):
    x_folder = r'/workspace/data/orCaScore/Challenge_data/Training_set/Images'
    y_folder = r'/workspace/data/cytran/net_B/cytran_pos_orca/images'
    
    for i in ids:
        x_path = os.path.join(x_folder, f"{i}{ct_type[moved]}.mhd")
        x_image = get_image_array(x_path)
        x_image = normalize_image(x_image)
        y_image = pkload(os.path.join(y_folder, f"{i}.pkl"))
        save_pkl(x_image, y_image, os.path.join(result_folder, f"{i}_NT.pkl"))

def transformed2transformed(ids, result_folder):
    x_folder = r'/workspace/data/cytran/net_A/cytran_pos_orca/images'
    y_folder = r'/workspace/data/cytran/net_B/cytran_pos_orca/images'
    
    for i in ids:
        x_image = pkload(os.path.join(x_folder, f"{i}.pkl"))
        y_image = pkload(os.path.join(y_folder, f"{i}.pkl"))
        save_pkl(x_image, y_image, os.path.join(result_folder, f"{i}_TT.pkl"))
    
def fuse_splits(folderA, folderB, result_folder, suffixA = "A", suffixB = "B"):
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    for filename in os.listdir(folderA):
        if filename.endswith('.pkl'):
            base_name = os.path.splitext(filename)[0]
            novo_nome = f"{base_name}_{suffixA}.pkl"
            origem = os.path.join(folderA, filename)
            destino = os.path.join(result_folder, novo_nome)
            # Move o arquivo
            shutil.copy(origem, destino)

    # Renomeie e mova os arquivos da pasta B para a pasta C
    for filename in os.listdir(folderB):
        if filename.endswith('.pkl'):
            # Extrai o nome do arquivo sem extensão
            base_name = os.path.splitext(filename)[0]
            # Cria o novo nome com o sufixo 'B.pkl'
            novo_nome = f"{base_name}_{suffixB}.pkl"
            # Caminho completo do arquivo de origem e de destino
            origem = os.path.join(folderB, filename)
            destino = os.path.join(result_folder, novo_nome)
            # Move o arquivo
            shutil.copy(origem, destino)
            
def copy_split(origin, destination):
    if not os.path.exists(destination):
        os.makedirs(destination)
    
    for filename in os.listdir(origin):
        if filename.endswith('.pkl'):
            origem = os.path.join(origin, filename)
            destino = os.path.join(destination, filename)
            shutil.copy(origem, destino)

def normalize_image(image):
    result = image.copy()
    result = result + 1024
    result[result < 0] = 0
    result = result / 1e3
    result = result - 1
    return result
        

    

def main():
    mode = 'T52_normalized'
    phase = 'train'
    result_folder = f'/workspace/data/orCaScore/vit-v-net/{mode}/{phase}'
    split_path = f'/workspace/data/orCaScore/cytran/{phase}_data.csv'
    
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    moved = 'ARTERIAL'
    fixed = 'NATIVE'
    ct_type = {
        'ARTERIAL': 'CTAI',
        'NATIVE': 'CTI'
    }
    
    ids = get_ids(split_path)
    
    # T26 normal
    match (mode):
        case 'T26_normal':
            normal2normal(ids, ct_type, moved, fixed, result_folder)
        case 'T26_normalized':
            normalized2normalized(ids, ct_type, moved, fixed, result_folder)
        case 'T26_transformed':
            transformed2normal(ids, ct_type, fixed, result_folder)
        case 'T52':
            fA = f'/workspace/data/orCaScore/vit-v-net/T26_normal/{phase}'
            fB = f'/workspace/data/orCaScore/vit-v-net/T26_transformed/{phase}'
            fuse_splits(fA, fB, result_folder, suffixA = "N", suffixB = "T")
        case 'T52_normalized':
            fA = f'/workspace/data/orCaScore/vit-v-net/T26_normalized/{phase}'
            fB = f'/workspace/data/orCaScore/vit-v-net/T26_transformed/{phase}'
            fuse_splits(fA, fB, result_folder, suffixA = "N", suffixB = "T")
        case 'T104':
            f52_folder = f'/workspace/data/orCaScore/vit-v-net/T52/{phase}'
            normal2transformed(ids, ct_type, moved, result_folder)
            transformed2transformed(ids, result_folder)
            copy_split(f52_folder, result_folder)
        case _:
            print("Invalid mode")
    
    

if __name__ == '__main__':
    main()