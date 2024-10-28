from PIL import Image
import os
from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os
from rdkit import Chem


# def smile2feature(data_root, type, file_data):
#     with open(file_data, "r") as f:
#         data_list = f.read().strip().split("\n")

#     """Exclude data contains '.' in the SMILES format."""  # The '.' represents multiple chemical molecules
#     data_list = [d for d in data_list if '.' not in d.strip().split()[0]]

#     smile_features = []
#     file_name = data_root + "/" + type + "_smile_features.txt"
#     with open(file_name, "w") as w:
#         for i, data in enumerate(data_list):
#             if i % 50 == 0:
#                 print('/'.join(map(str, [i + 1, len(data_list)])))

#             fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
#             factory = ChemicalFeatures.BuildFeatureFactory(fdefName)
#             smile = data.strip().split(" ")[0]
#             mol = Chem.MolFromSmiles(smile)
#             feats = factory.GetFeaturesForMol(mol)

#             line = ""
#             for f in feats:
#                 s = str(f.GetFamily())
#                 s += " " + str(f.GetType())
#                 s += " " + str(f.GetAtomIds())
#                 # s += " " + str(f.GetId())
#                 line += s + " "
#             line += "\n"
#             w.write(line)



def smile2feature(data_root, type, file_data):
    with open(file_data, "r") as f:
        data_list = f.read().strip().split("\n")

    # Exclude data containing '.' in the SMILES format.
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]

    smile_features = []
    file_name = data_root + "/" + type + "_smile_features.txt"
    fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    factory = ChemicalFeatures.BuildFeatureFactory(fdefName)

    with open(file_name, "w") as w:
        for i, data in enumerate(data_list):
            if i % 50 == 0:
                print('/'.join(map(str, [i + 1, len(data_list)])))

            smile = data.strip().split(" ")[0]

            try:
                # Try to generate the molecule from SMILES
                mol = Chem.MolFromSmiles(smile)
                if mol is None:
                    raise ValueError(f"Invalid SMILES string: {smile}")

                # Try to extract features from the molecule
                feats = factory.GetFeaturesForMol(mol)
                if not feats:
                    raise ValueError(f"No features found for molecule: {smile}")

                # Build feature string
                line = ""
                for f in feats:
                    s = str(f.GetFamily()) + " " + str(f.GetType()) + " " + str(f.GetAtomIds())
                    line += s + " "
                line += "\n"

                # Write features to file
                w.write(line)

            except Exception as e:
                # Log or print the error message and skip the invalid molecule
                print(f"Error processing SMILES {smile}: {e}")

if __name__ == '__main__':
    # dataset_name = "Davis"
    dataset_name = "BioSNAP"
    data_root = "data/" + dataset_name
    train_file = data_root + "/" + dataset_name + "_train.txt"
    test_file = data_root + "/" + dataset_name + "_test.txt"
    val_file = data_root + "/" + dataset_name + "_val.txt"

    smile2feature(data_root, "train", train_file)
    smile2feature(data_root, "val", val_file)
    smile2feature(data_root, "test", test_file)
