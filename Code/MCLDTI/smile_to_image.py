from PIL import Image
import os
from rdkit.Chem import Draw
from rdkit import Chem

def smile2pic(file_path, file_data):
    with open(file_data, "r") as f:
        data_list = f.read().strip().split("\n")

    # Exclude data that contains '.' in the SMILES format.
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]

    smiles = []
    for i, data in enumerate(data_list):
        if i % 100 == 0:
            print('/'.join(map(str, [i + 1, len(data_list)])))

        smile = data.strip().split(" ")[0]

        try:
            # Convert SMILES to molecule
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                raise ValueError(f"Invalid SMILES string: {smile}")
            
            # Generate canonical SMILES
            canonical_smi = Chem.MolToSmiles(mol)
            canonical_mol = Chem.MolFromSmiles(canonical_smi)

            if canonical_mol is None:
                raise ValueError(f"Could not generate canonical molecule for SMILES: {smile}")

            # Generate image
            img = Draw.MolToImage(canonical_mol, size=(pic_size, pic_size), wedgeBonds=False)
            number = str(i + 1).zfill(len(str(len(data_list))))

            smiles.append(smile)

            save_name = file_path + "/" + number + ".png"
            img.save(save_name)

        except Exception as e:
            # Log the error and skip invalid SMILES
            print(f"Error processing SMILES {smile}: {e}")

def pic_info(file_path):
    file_list = os.listdir(file_path)
    num = 0
    for pic in file_list:
        if ".png" in pic:
            num += 1
    str_len = len(str(num))
    print(str_len)
    print(file_path)
    with open(file_path + "/img_inf_data", "w") as f:
        for i in range(num):
            number = str(i + 1).zfill(len(str(len(file_list))))
            if i == num - 1:
                f.write(file_path + "/" + number + ".png" + "\t" + number + ".png")
            else:
                f.write(file_path + "/" + number + '.png' + "\t" + number + '.png' + "\n")

if __name__ == '__main__':
    # dataset_name = "BindingDB"
    dataset_name = "BioSNAP"
    pic_size = 256
    data_root = "data/" + dataset_name
    train_file = data_root + "/" + dataset_name + "_train.txt"
    test_file = data_root + "/" + dataset_name + "_test.txt"
    val_file = data_root + "/" + dataset_name + "_val.txt"

    train_path = data_root + "/train/"
    if not os.path.exists(train_path):
        os.makedirs(train_path)

    test_path = data_root + "/test/"
    if not os.path.exists(test_path):
        os.makedirs(test_path)

    val_path = data_root + "/val/"
    if not os.path.exists(val_path):
        os.makedirs(val_path)

    pic_train_path = train_path + "Img_" + str(pic_size) + "_" + str(pic_size)
    if not os.path.exists(pic_train_path):
        os.makedirs(pic_train_path)

    pic_test_path = test_path + "Img_" + str(pic_size) + "_" + str(pic_size)
    if not os.path.exists(pic_test_path):
        os.makedirs(pic_test_path)

    pic_val_path = val_path + "Img_" + str(pic_size) + "_" + str(pic_size)
    if not os.path.exists(pic_val_path):
        os.makedirs(pic_val_path)

    smile2pic(pic_train_path, train_file)
    print("Train_Pic generated. size=", pic_size, "*", pic_size, "----")

    smile2pic(pic_test_path, test_file)
    print("Test_Pic generated. size=", pic_size, "*", pic_size, "----")

    smile2pic(pic_val_path, val_file)
    print("Val_Pic generated. size=", pic_size, "*", pic_size, "----")

    pic_info(pic_train_path)
    pic_info(pic_test_path)
    pic_info(pic_val_path)
