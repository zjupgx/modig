import os
import numpy as np
import pandas as pd
from sklearn import preprocessing

snv = pd.read_csv('F:/Omics-data/TCGA-PANCAN/TCGA-PANCAN_snv_mc3_cancer.tsv',sep='\t',index_col=0)
cnv = pd.read_csv('F:/Omics-data/TCGA-PANCAN/TCGA-PANCAN_cnv_cancer.tsv',sep='\t',index_col=0)
mrna = pd.read_csv('F:/Omics-data/TCGA-PANCAN/TCGA-PANCAN_mrna_fold_change_cancer.tsv',sep='\t',index_col=0)
dm = pd.read_csv('F:/Omics-data/TCGA-PANCAN/TCGA-PANCAN_dm_fold_change_cancer.tsv',sep='\t',index_col=0)

gene = pd.read_csv("F:/Omics-data/TCGA-PANCAN/gene_info_for_GOSemSim.csv")
gene_list = list(set(gene['Symbol']))


genelist = list(set(snv.index)|set(cnv.index)|set(mrna.index)|set(dm.index)|set(gene_list))
print(len(genelist))
temp = pd.DataFrame(index=genelist,columns=mrna.columns)
print(temp.shape)

snv_adj = temp.combine_first(snv).loc[gene_list].sort_index()
cnv_adj = temp.combine_first(cnv).loc[gene_list].sort_index()
mrna_adj = temp.combine_first(mrna).loc[gene_list].sort_index()
dm_adj = temp.combine_first(dm).loc[gene_list].sort_index()

def omics_preprocess(use_quantile_norm,snv,cnv,mrna,dm):
    if use_quantile_norm:
        scaler = preprocessing.QuantileTransformer(output_distribution='normal')
        #scaler = preprocessing.StandardScaler()
        mrna_norm = preprocessing.MinMaxScaler().fit_transform(np.abs(scaler.fit_transform(mrna)))
        snv_norm = preprocessing.MinMaxScaler().fit_transform(snv)
        dm_norm = preprocessing.MinMaxScaler().fit_transform(np.abs(scaler.fit_transform(dm)))
        cnv_norm = preprocessing.MinMaxScaler().fit_transform(np.abs(scaler.fit_transform(cnv)))
    else:
        scaler = preprocessing.MinMaxScaler()
        mrna_norm = scaler.fit_transform(np.abs(mrna))
        snv_norm = scaler.fit_transform(snv)
        dm_norm = scaler.fit_transform(np.abs(dm))
        cnv_norm = scaler.fit_transform(np.abs(cnv))

    multi_omics_features = np.concatenate((snv_norm, dm_norm, mrna_norm, cnv_norm), axis=1)
    
    return multi_omics_features


use_quantile_norm ='False'
multi_omics_features = omics_preprocess(use_quantile_norm,snv_adj,cnv_adj,mrna_adj,dm_adj)

cancer_type_list = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM',
       'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC',
       'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD','TGCT', 'THCA', 'THYM', 'UCEC',
       'UCS', 'UVM']

omic_list = []
for omic in ['MF','CNV','GE','METH']:
    d = [': '.join([omic,x]) for x in cancer_type_list]
    omic_list.extend(d)

multi_omics_features_df = pd.DataFrame(multi_omics_features, index=snv_adj.index,columns=omic_list)
multi_omics_features_df.fillna(0,inplace=True)
multi_omics_features_df.to_csv('F:/Omics-data/TCGA-PANCAN/biological_features.tsv',sep='\t')