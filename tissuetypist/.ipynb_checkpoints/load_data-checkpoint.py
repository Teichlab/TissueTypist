import pkg_resources
import io
import joblib

def key_tissue_genes():
    '''
    Load the key genes (for tissue identification) distributed with the package.
    
    Returns a list of genes.
    '''
    #this picks up the pickle shipped with the package
    stream = pkg_resources.resource_stream(__name__, 'key_tissue_genes.txt')
    # Convert the binary stream to a text stream
    text_stream = io.TextIOWrapper(stream, encoding='utf-8')
    key_tissue_genes = [line.strip() for line in text_stream]
    return key_tissue_genes

def trained_models_full():
    '''
    Load the the trained model (for a full transcriptome dataset) distributed with the package.
    
    Returns a dictionary of trained models.
    '''
    pipeline_list = ['weight2neighbours-0.0_weight2edge-0_pipeline.joblib',
                     'weight2neighbours-0.3_weight2edge-5_pipeline.joblib',
                     'weight2neighbours-1.0_weight2edge-5_pipeline.joblib'
                    ]
    # this picks up the models shipped with the package
    # get trained pipelines
    model_dict={}
    for file in pipeline_list:
        pipeline_name = file.replace('_pipeline.joblib','')
        path_to_pipeline = f"trained_models_full/{file}"
        stream = pkg_resources.resource_stream(__name__, path_to_pipeline)
        model_dict[pipeline_name]=joblib.load(stream)
    return model_dict
    