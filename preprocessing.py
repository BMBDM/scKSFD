from anndata import AnnData
import scvelo as scv
import pandas as pd 

def preprocess_RNASeq(df):
    adata = AnnData(df.T)
    scv.pp.filter_genes_dispersion(adata, n_top_genes=2000)
    scv.pp.log1p(adata)
    df0 = pd.DataFrame(adata.X.T, index=adata.var['highly_variable'].index, columns=df.columns)
    return df0

