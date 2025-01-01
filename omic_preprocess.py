import pandas as pd
from sklearn import preprocessing
import numpy as np

# 读取单核苷酸变异（SNV）数据文件，以制表符'\t'作为分隔符，将文件的第一列作为索引列（index_col=0），并将数据存储在snv变量中
snv = pd.read_csv('F:/Omics-data/TCGA-PANCAN/TCGA-PANCAN_snv_mc3_cancer.tsv', sep='\t', index_col=0)
# 读取拷贝数变异（CNV）数据文件，同样以'\t'作为分隔符，第一列作为索引列，存储到cnv变量
cnv = pd.read_csv('F:/Omics-data/TCGA-PANCAN/TCGA-PANCAN_cnv_cancer.tsv', sep='\t', index_col=0)
# 读取信使核糖核酸（mRNA）表达量变化数据文件，按'\t'分隔，以第一列做索引列，保存到mrna变量
mrna = pd.read_csv('F:/Omics-data/TCGA-PANCAN/TCGA-PANCAN_mrna_fold_change_cancer.tsv', sep='\t', index_col=0)
# 读取DNA甲基化（DM）变化数据文件，按'\t'分隔，以第一列做索引列，存入dm变量
dm = pd.read_csv('F:/Omics-data/TCGA-PANCAN/TCGA-PANCAN_dm_fold_change_cancer.tsv', sep='\t', index_col=0)

# 读取包含基因信息的文件，用于后续获取基因名称列表等操作
gene = pd.read_csv("F:/Omics-data/TCGA-PANCAN/gene_info_for_GOSemSim.csv")
# 从基因信息文件的'Symbol'列提取所有不重复的基因名称，组成列表gene_list，这些基因名称后续会用于数据对齐等处理
gene_list = list(set(gene['Symbol']))

# 通过对SNV、CNV、mRNA、DM这四种组学数据的索引（通常代表基因）取并集，再结合gene_list中的基因，得到一个包含所有涉及的不重复基因的列表genelist
# 这样能保证后续处理的数据是基于相同的基因集合进行的，避免因基因不一致导致的问题
genelist = list(set(snv.index) | set(cnv.index) | set(mrna.index) | set(dm.index) | set(gene_list))
# 打印genelist的长度，即共同基因的数量，方便了解数据规模和后续处理的数据量大小情况
print(len(genelist))

# 创建一个以genelist为行索引、以mrna的列名为列名的空DataFrame对象temp，它将作为一个基础框架，用于后续合并不同组学数据
temp = pd.DataFrame(index=genelist, columns=mrna.columns)
# 打印temp的形状，展示其行数（对应基因数量）和列数（对应mRNA数据的列数）情况，便于直观查看数据结构的大致样子
print(temp.shape)

# 使用combine_first方法将snv数据合并到temp中，该方法会优先保留snv中的数据，如果temp对应位置没有值则用snv中的值填充
# 然后通过loc方法筛选出只包含gene_list中基因的数据，并按照索引进行排序（sort_index），得到对齐到相同基因集合且索引有序的snv_adj数据
snv_adj = temp.combine_first(snv).loc[gene_list].sort_index()
# 与上述处理snv的操作类似，对cnv数据进行合并、筛选和排序，得到处理后的cnv_adj数据，使其与其他组学数据基于相同基因集合对齐
cnv_adj = temp.combine_first(cnv).loc[gene_list].sort_index()
# 同样的方式处理mRNA数据，得到mrna_adj，确保其基因范围与其他组学数据一致且索引有序
mrna_adj = temp.combine_first(mrna).loc[gene_list].sort_index()
# 对DM数据进行相同的合并、筛选和排序操作，得到dm_adj，以便后续进行统一的处理和分析
dm_adj = temp.combine_first(dm).loc[gene_list].sort_index()

# 定义一个名为omics_preprocess的函数，用于对输入的多种组学数据进行预处理操作，
# 预处理操作包括标准化、归一化等，目的是使不同组学数据具有相似的数值范围，便于后续整合和分析
def omics_preprocess(use_quantile_norm, snv, cnv, mrna, dm):
    """
    参数:
    - use_quantile_norm: 一个布尔值（虽然代码中传入的是字符串，实际使用时最好转换为布尔类型更严谨），用于决定是否使用分位数标准化方法。
    - snv、cnv、mrna、dm: 分别为经过前面初步处理后，基于相同基因集合对齐的单核苷酸变异、拷贝数变异、信使核糖核酸、DNA甲基化这四种组学数据。
    """
    # 如果use_quantile_norm为True，表示使用分位数标准化方法
    if use_quantile_norm:
        # 创建一个QuantileTransformer对象，用于将数据转换为正态分布（output_distribution='normal'），
        # 这有助于使数据分布更符合一些模型的假设，提高模型性能（此处有一行被注释掉的代码，可能原本考虑过其他标准化方式）
        scaler = preprocessing.QuantileTransformer(output_distribution='normal')
        #scaler = preprocessing.StandardScaler()

        # 对于mRNA数据，先使用scaler（QuantileTransformer）进行转换，再取绝对值，最后使用MinMaxScaler将数据归一化到[0, 1]区间
        mrna_norm = preprocessing.MinMaxScaler().fit_transform(np.abs(scaler.fit_transform(mrna)))
        # 对于SNV数据，直接使用MinMaxScaler将其归一化到[0, 1]区间（先取绝对值，因为数据本身的特点可能需要这样处理）
        snv_norm = preprocessing.MinMaxScaler().fit_transform(snv)
        # 对于DM数据，同样先使用scaler进行转换（如果use_quantile_norm为True），取绝对值后用MinMaxScaler归一化到[0, 1]区间
        dm_norm = preprocessing.MinMaxScaler().fit_transform(np.abs(scaler.fit_transform(dm)))
        # 对于CNV数据，也是先取绝对值，再用MinMaxScaler进行归一化操作，使其数值范围在[0, 1]区间内
        cnv_norm = preprocessing.MinMaxScaler().fit_transform(np.abs(scaler.fit_transform(cnv)))
    # 如果use_quantile_norm为False，不使用分位数标准化，直接使用MinMaxScaler对各数据进行归一化
    else:
        scaler = preprocessing.MinMaxScaler()
        # 对mRNA数据取绝对值后，使用MinMaxScaler进行归一化，将其数值映射到[0, 1]区间
        mrna_norm = scaler.fit_transform(np.abs(mrna))
        # 对SNV数据直接使用MinMaxScaler进行归一化
        snv_norm = scaler.fit_transform(snv)
        # 对DM数据取绝对值后用MinMaxScaler进行归一化
        dm_norm = scaler.fit_transform(np.abs(dm))
        # 对CNV数据取绝对值后用MinMaxScaler进行归一化
        cnv_norm = scaler.fit_transform(np.abs(cnv))

    # 使用np.concatenate函数沿着列方向（axis=1）将经过预处理后的四种组学数据（SNV、DM、mRNA、CNV）拼接起来，
    # 形成一个包含所有组学特征的二维数组multi_omics_features，这样每个基因对应的多组学特征就整合到了一起
    multi_omics_features = np.concatenate((snv_norm, dm_norm, mrna_norm, cnv_norm), axis=1)

    # 返回整合后的多组学特征数据，以便后续进一步处理或使用
    return multi_omics_features


# 将字符串'False'赋值给use_quantile_norm变量（如前面所述，更严谨的做法是转换为布尔值False），用于决定采用何种预处理方式
use_quantile_norm = 'False'
# 调用omics_preprocess函数，传入use_quantile_norm以及前面整理好的四种组学数据（snv_adj、cnv_adj、mrna_adj、dm_adj），
# 获取经过预处理和整合后的多组学特征数据，并存储在multi_omics_features变量中
multi_omics_features = omics_preprocess(use_quantile_norm, snv_adj, cnv_adj, mrna_adj, dm_adj)

# 定义一个包含多种癌症类型名称的列表，这些癌症类型在后续构建数据列名时会用到，用于标识不同组学特征与癌症类型的关联
cancer_type_list = ['ACC', 'BLCA', 'BRCA', 'CESC', 'CHOL', 'COAD', 'DLBC', 'ESCA', 'GBM',
                    'HNSC', 'KICH', 'KIRC', 'KIRP', 'LAML', 'LGG', 'LIHC', 'LUAD', 'LUSC',
                    'MESO', 'OV', 'PAAD', 'PCPG', 'PRAD', 'READ', 'SARC', 'SKCM', 'STAD',
                    'TGCT', 'THCA', 'THYM', 'UCEC',
                    'UCS', 'UVM']

omic_list = []
# 循环遍历['MF', 'CNV', 'GE', 'METH']列表中的每个元素（可能代表不同组学类型的缩写，具体含义需结合实际背景确定）
for omic in ['MF', 'CNV', 'GE', 'METH']:
    # 对于每个组学类型缩写，将其与cancer_type_list中的每个癌症类型名称进行拼接，中间用': '隔开，
    # 例如 'MF: ACC'、'MF: BLCA'等，形成新的字符串列表d，用于清晰标识每个组学特征对应的组学类型和癌症类型
    d = [': '.join([omic, x]) for x in cancer_type_list]
    # 将d列表中的元素添加到omic_list中，最终omic_list包含了所有用于构建多组学特征数据列名的字符串
    omic_list.extend(d)

# 使用pandas的DataFrame构造函数，将multi_omics_features二维数组转换为DataFrame数据结构，
# 指定行索引为snv_adj的索引（也就是前面确定的共同基因索引），列名为omic_list中的字符串，得到整合后的多组学特征数据框
multi_omics_features_df = pd.DataFrame(multi_omics_features, index=snv_adj.index, columns=omic_list)
# 将multi_omics_features_df中可能存在的缺失值填充为0，确保数据完整性，直接在原数据框上进行修改（inplace=True）
multi_omics_features_df.fillna(0, inplace=True)
# 使用to_csv方法将这个包含多组学特征的数据框保存为一个以制表符（'\t'）为分隔符的文件biological_features.tsv，
# 存储在指定的文件路径'F:/Omics-data/TCGA-PANCAN/'下，方便后续读取该文件进行进一步的数据分析、机器学习建模等操作
multi_omics_features_df.to_csv('F:/Omics-data/TCGA-PANCAN/biological_features.tsv', sep='\t')