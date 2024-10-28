from PIL import Image
import os
from rdkit.Chem import Draw
from rdkit import Chem
import torch
import pandas as pd
import torch_geometric.transforms as T
import numpy as np

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


def split_text_data(file_path, train_indices, val_indices, test_indices,
                    train_file='train.txt', val_file='val.txt', test_file='test.txt'):
        # Load the data
        data = pd.read_csv(file_path, sep='\t', header=None, names=['SMILES', 'SEQUENCE', 'LABEL'])

        # Use the indices to split data
        train_data = data.iloc[train_indices]
        val_data = data.iloc[val_indices]
        test_data = data.iloc[test_indices]

        # Save to separate files
        train_data.to_csv(train_file, sep='\t', header=False, index=False)
        val_data.to_csv(val_file, sep='\t', header=False, index=False)
        test_data.to_csv(test_file, sep='\t', header=False, index=False)

        print(f"Data split into:\nTrain: {len(train_data)} samples\nValidation: {len(val_data)} samples\nTest: {len(test_data)} samples")

if __name__ == '__main__':
    # dataset_name = "BindingDB"
    dataset_name = "BIOSNAP"


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'

    path = f'../../Data/{dataset_name}/hetero_data_biosnap.pt'
    data = torch.load(path)
    data = T.ToUndirected()(data)



    import random
    random.seed(42)
    torch.manual_seed(42)
    # del data['protein', 'rev_interaction', 'drug'].edge_label 
    transform = T.RandomLinkSplit(
        num_val=0.1,
        num_test=0.2,
        is_undirected=True,
        disjoint_train_ratio=0.2,
        neg_sampling_ratio=2.0,
        add_negative_train_samples=True,
        edge_types=("drug", "interaction", "protein"),
        rev_edge_types=("protein", "rev_interaction", "drug"), 
        split_labels=False
    )

    train_data, val_data, test_data = transform(data)
    # print('*'*100)
    # print('train data:', train_data)
    # print('*'*100)
    # print('val data:', val_data)
    # print('*'*100)
    # print('test data:', test_data)
    # print('*'*100)

    # Extract edge indices for train, val, and test sets
    # train_edge_index = train_data['drug', 'interaction', 'protein'].edge_index.cpu()
    # val_edge_index = val_data['drug', 'interaction', 'protein'].edge_index.cpu()
    # test_edge_index = test_data['drug', 'interaction', 'protein'].edge_index.cpu()

    
    train_labels = train_data[('drug','interaction','protein')].edge_label.unsqueeze(dim=-1).cpu().numpy()
    # print(train_labels.shape)
    val_labels = val_data[('drug','interaction','protein')].edge_label.unsqueeze(dim=-1).cpu().numpy()
    test_labels = test_data[('drug','interaction','protein')].edge_label.unsqueeze(dim=-1).cpu().numpy()

    with open( "../../Data/" + dataset_name + "/input/" + dataset_name + "_train_interactions.npy", 'wb') as f: 
        np.save(f,train_labels)
    with open( "../../Data/" + dataset_name + "/input/" + dataset_name + "_val_interactions.npy", 'wb') as f: 
        np.save(f,val_labels)
    with open( "../../Data/" + dataset_name + "/input/" + dataset_name + "_test_interactions.npy", 'wb') as f: 
        np.save(f,test_labels)
    



    # # Load text data for correlation
    # text_data = pd.read_csv(f'../../Data/{dataset_name}/smile_sequence.txt', sep=',', header=None, names=['SMILES', 'SEQUENCE', 'LABEL'])

    # # Map edge indices to rows in the text data
    # # (Assuming each edge corresponds directly to a row in text_data)
    # train_indices = train_edge_index[1].numpy()  # Get the target nodes as indices
    # val_indices = val_edge_index[1].numpy()
    # test_indices = test_edge_index[1].numpy()

    # # Create splits for text data
    # train_text_data = text_data.iloc[train_indices]
    # val_text_data = text_data.iloc[val_indices]
    # test_text_data = text_data.iloc[test_indices]

    # pic_size = 256
    # data_root = "../../Data/" + dataset_name

    # # Save to separate files
    # train_text_data.to_csv(data_root + "/" + dataset_name + "_train.txt", sep='\t', header=False, index=False)
    # val_text_data.to_csv(data_root + "/" + dataset_name + "_test.txt", sep='\t', header=False, index=False)
    # test_text_data.to_csv(data_root + "/" + dataset_name + "_val.txt", sep='\t', header=False, index=False)

    # print("Text data split and saved to train.txt, val.txt, and test.txt.")
    # print('*'*100)

    

    

    # train_file = data_root + "/" + dataset_name + "_train.txt"
    # test_file = data_root + "/" + dataset_name + "_test.txt"
    # val_file = data_root + "/" + dataset_name + "_val.txt"

    # train_path = data_root + "/train/"
    # if not os.path.exists(train_path):
    #     os.makedirs(train_path)

    # test_path = data_root + "/test/"
    # if not os.path.exists(test_path):
    #     os.makedirs(test_path)

    # val_path = data_root + "/val/"
    # if not os.path.exists(val_path):
    #     os.makedirs(val_path)

    # pic_train_path = train_path + "Img_" + str(pic_size) + "_" + str(pic_size)
    # if not os.path.exists(pic_train_path):
    #     os.makedirs(pic_train_path)

    # pic_test_path = test_path + "Img_" + str(pic_size) + "_" + str(pic_size)
    # if not os.path.exists(pic_test_path):
    #     os.makedirs(pic_test_path)

    # pic_val_path = val_path + "Img_" + str(pic_size) + "_" + str(pic_size)
    # if not os.path.exists(pic_val_path):
    #     os.makedirs(pic_val_path)

    # smile2pic(pic_train_path, train_file)
    # print("Train_Pic generated. size=", pic_size, "*", pic_size, "----")

    # smile2pic(pic_test_path, test_file)
    # print("Test_Pic generated. size=", pic_size, "*", pic_size, "----")

    # smile2pic(pic_val_path, val_file)
    # print("Val_Pic generated. size=", pic_size, "*", pic_size, "----")

    # pic_info(pic_train_path)
    # pic_info(pic_test_path)
    # pic_info(pic_val_path)
