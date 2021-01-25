import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import ast

from IPython import embed


def convert_np_to_mx(numpy_of_strings):
    aux = [np.fromstring(x, dtype=int, sep=' ') for x in numpy_of_strings]
    return np.asmatrix(aux)


def intersection(lst1, lst2):
    if not isinstance(lst1, list):
        lst1 = [lst1]
    if not isinstance(lst2, list):
        lst2 = [lst2]
    try:
        lst3 = [value for value in lst1 if value in lst2]
        return lst3
    except:
        embed()


def load_matrices(dataset_path):
    drug_dis = np.loadtxt(f"{dataset_path}mat_drug_disease.txt", dtype='str', delimiter='\n')  # 708, 11.205
    drug_prot = np.loadtxt(f"{dataset_path}mat_drug_protein.txt", dtype='str', delimiter='\n')  # 708, 3.023
    # aux = np.loadtxt(f"{dataset_path}mat_drug_protein_remove_homo.txt", dtype='str', delimiter='\n')
    prot_dis = np.loadtxt(f"{dataset_path}mat_protein_disease.txt", dtype='str', delimiter='\n')  # 1.512, 11.205

    # prot_drug = np.loadtxt(f"{dataset_path}mat_protein_drug.txt", dtype='str', delimiter='\n')  # 1.512, 1.415

    drug_dis = convert_np_to_mx(drug_dis)  # 708, 5.603
    drug_prot = convert_np_to_mx(drug_prot)  # 708, 1.512
    prot_dis = convert_np_to_mx(prot_dis)  # 1.512, 5.603
    # prot_drug = convert_np_to_mx(prot_drug)  # 1.512, 708

    return drug_dis, drug_prot, prot_dis


if __name__ == '__main__':

    data_path = f'./data/drugs/'

    if not os.path.exists(f'{data_path}train_data.csv'):
        
        if not os.path.exists(f'{data_path}adjacency_mx.npy'):
            # BUILD ADJACENCY MATRIX

            drug_dis, drug_prot, prot_dis = load_matrices(data_path)

            # mo = np.multiply.outer
            drugs = drug_dis.shape[0]
            proteins = prot_dis.shape[0]
            diseases = drug_dis.shape[1]

            # IDEA: 1. EXTEND DRUG_DIS
            drug_drug = np.zeros((drugs, drugs))
            diseases_diseases = np.zeros((diseases, diseases))
            protein_protein = np.zeros((proteins, proteins))
            # IDEA: 2. MAKE IT SYMMETRIC
            # AXIS = 1  IS TO CONCATENATE NEXT
            drug_dis_ext = np.concatenate((drug_drug, drug_dis), axis=1)  # (708, 6311)
            drug_dis_T_extended = np.concatenate((drug_dis.T, diseases_diseases), axis=1)  # (5603, 6311)
            # AXIS = 0  IS TO APPEND BELOW
            ADJ = np.concatenate((drug_dis_ext, drug_dis_T_extended), axis=0)  # (6311, 6311)
            # IDEA: 3. ADD PROTEIN (CONTEXT)
            column_mx = np.concatenate((drug_prot, prot_dis.T), axis=0)   # (6311, 1512)
            rows_mx = np.concatenate((drug_prot.T, prot_dis), axis=1)  # (1512, 6311)
            # append zeros protein protein
            column_mx = np.concatenate((column_mx, protein_protein), axis=0)  # (7823, 1512)
            # merge all
            ADJ = np.concatenate((ADJ, rows_mx), axis=0)  # (7823, 6311)
            ADJ = np.concatenate((ADJ, column_mx), axis=1)  # (7823, 7823)

            assert ADJ.shape[0] == ADJ.shape[1] == (drugs + proteins + diseases)

            np.save(f'{data_path}adjacency_mx.npy', ADJ)
            a = ADJ
        else:
            a = np.load(f'{data_path}adjacency_mx.npy')

        # BUILD TRAINING DATA
        drugs = 708
        proteins = 1512
        diseases = 5603

        # for user in range(drugs):
        # user = 0
        # all_idx = np.where(a[user] == 1)[0]
        # assert len(all_idx[all_idx < drugs]) == 0
        # idx_items = all_idx[all_idx < (diseases + drugs)]
        # idx_context = all_idx[all_idx >= (diseases + drugs)]
        drug_dis, drug_prot, prot_dis = load_matrices(data_path)

        drug_prot  # (708, 1512)
        dis_prot = prot_dis.T  # (5603, 1512)

        drugs_si = pd.DataFrame(columns=('drug', 'proteins_drugs'))
        for i, row in tqdm(enumerate(drug_prot), total=len(drug_prot)):
            drugs_si.loc[i] = [i, [i for i, e in enumerate(row.tolist()[0]) if e == 1]]

        diseases_si = pd.DataFrame(columns=('disease', 'proteins_disease'))
        for i, row in tqdm(enumerate(dis_prot), total=len(dis_prot)):
            diseases_si.loc[i] = [i, [i for i, e in enumerate(row.tolist()[0]) if e == 1]]

        drugs_si = drugs_si.mask(drugs_si.applymap(str).eq('[]'))
        diseases_si = diseases_si.mask(diseases_si.applymap(str).eq('[]'))

        drugs_si = drugs_si.fillna(proteins)
        diseases_si = diseases_si.fillna(proteins)

        #  GENERATE DB USER-ITEM
        df = pd.DataFrame(columns=('drug', 'disease'))
        for i, row in tqdm(enumerate(drug_dis), total=len(drug_dis), desc='generate db...'):
            diseases_related = [i for i, e in enumerate(row.tolist()[0]) if e == 1]
            for j in diseases_related:
                df = df.append({'drug': i, 'disease': j}, ignore_index=True)

        df['disease'] = df['disease'] + drugs
        diseases_si['disease'] = diseases_si['disease'] + drugs
        # drugs_si.rename(columns={"protein": "proteins_drug"}, inplace=True)
        # diseases_si.rename(columns={"protein": "proteins_disease"}, inplace=True)

        df = pd.merge(df, drugs_si, how="left", on=["drug"])
        df = pd.merge(df, diseases_si, how="left", on=["disease"])

        df.to_csv(f'{data_path}train_data.csv')
    else:
        drugs = 708
        proteins = 1512
        diseases = 5603
        
        df = pd.read_csv(f'{data_path}train_data.csv', index_col=0)
        context_list = []
        # proteins = 1512 == unknown context
        for i, row in tqdm(df.iterrows(), total=len(df), desc='GENERATE INTERSECTION CONTEXT...'):
            # row = df.iloc[0]
            lst1 = ast.literal_eval(row['proteins_drug'])
            lst2 = ast.literal_eval(row['proteins'])
            context_list.append(intersection(lst1, lst2))

        df = df.assign(context=context_list)

        count_nontype = 0
        for lst in df['context']:
            if not type(lst) == np.float and proteins in lst and lst:
                count_nontype += 1

        df = df.mask(df.applymap(str).eq('[]'))
        df = df.mask(df.applymap(str).eq('[1512]'))

        print('DF HAS 71808/199214 NULL interactions')

        # CREATE DB JUST WITH CONTEXT
        df1 = df.dropna()
        df1.to_csv(f'{data_path}train_data_allcontext.csv')

        # CREATE DB FILLING UP EMPTY CONTEXT
        df2 = df.fillna(proteins)
        df2.to_csv(f'{data_path}train_data_allcontext_PLUSfaked.csv')
        print('ALL DONE AND CSV SAVED')
        


