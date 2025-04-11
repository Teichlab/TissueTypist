import pkg_resources
import scanpy as sc
import io

def reference_adata():
    '''
    Load the reference data (collection of VisiumSD data) distributed with the package.
    
    Returns a anndata.
    '''
    #this picks up the pickle shipped with the package
    stream = pkg_resources.resource_stream(__name__, 'visiumsd_oct_raw.h5ad')
    adata_ref = sc.read_h5ad(stream)
    return adata_ref

def key_tissue_genes():
    '''
    Load the key genes for tissue identification distributed with the package.
    
    Returns a list of genes.
    '''
    #this picks up the pickle shipped with the package
    stream = pkg_resources.resource_stream(__name__, 'key_tissue_genes.txt')
    # Convert the binary stream to a text stream
    text_stream = io.TextIOWrapper(stream, encoding='utf-8')
    key_tissue_genes = [line.strip() for line in text_stream]
    return key_tissue_genes