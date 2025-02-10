#%%- imports
from model import Segmentor
from loguru import logger
import torch
#%% - define hparams
class HParams:
    wav_path= ''
    dataset='timit' #choices=['timit', 'buckeye']
    run_dir='/tmp/segmentation' #'directory for saving run outputs (logs, ckpt, etc.)'
    exp_name='segmentation_experiment' #help='experiment name')
    load_ckpt=None #'path to a pre-trained model, if provided, training will resume from that point')
    gpus='-1' #help='gpu ids to use, -1 for cpu')
    devrun=False #action='store_true', help='dev run on a dataset of size `devrun_size`')
    devrun_size=10 #help='size of dataset for dev run')
    test=False #action='store_true', help='flag to indicate to run a test epoch only (not training will take place)')

    lr=0.001 #help='initial learning rate')
    optimizer='adam'
    momentum=0.9
    epochs=50 #help='upper epoch limit')
    batch_size=2 #metavar='N', help='batch size')
    dropout=0.0 #help='dropout probability value')
    seed=1245 #help='random seed')
    patience=5 #help='patience for early stopping')
    gamma=0.15 #help='gamma margin')
    overfit=-1 #help='gamma margin')
    val_percent_check=1.0 #help='how much of the validation set to check')
    val_check_interval=1.0 #help='validation check every K epochs')
    val_ratio=0.1 #help='precentage of validation from train')

    rnn_input_size=50 #help='number of inputs')
    rnn_dropout=0.3 #help='dropout')
    rnn_hidden_size=200 #help='RNN hidden layer size')
    birnn=True, #help='BILSTM, if define will be biLSTM')
    rnn_layers=2 #help='number of lstm layers')
    min_seg_size=1 #help='minimal size of segment, examples with segments smaller than this will be ignored')
    max_seg_size=100 #help='see `min_seg_size`')
    max_len=500 #help='maximal size of sequences')
    feats="mfcc" #choices=["mfcc", "mel", "spect"], help='type of acoustic features to use')
    random_trim=False #action="store_true", help='if this flag is on seuqences will be randomly trimmed')
    delta_feats=False #action="store_true", help='if this flag is on delta features will be added')
    dist_feats=False #action="store_true", help='if this flag is on the euclidean features will be added (see paper)')
    normalize=False #action="store_true", help='flag to normalize features')
    bin_cls=0 #type=float, help='coefficient of binary classification loss')
    phn_cls=0 #type=float, help='coefficient of phoneme classification loss')
    n_fft=160 #help='n_fft for feature extraction')
    hop_length=160 #help='hop_length for feature extraction')
    n_mels=40 #help='number of mels')
    n_mfcc=13 #help='number of mfccs')
    
    cuda = torch.cuda.is_available()
    
    n_classes = {'timit': 39,'buckeye': 40}[dataset]

#%%
def build_model(config):
    config = HParams()
    segmentor = Segmentor(config)

    if config.load_ckpt not in [None, "None"]:
        logger.info(f"loading checkpoint from: {config.load_ckpt}")
        model_dict = segmentor.state_dict()
        weights = torch.load(config.load_ckpt, map_location='cuda:0')["state_dict"]
        weights = {k.replace("segmentor.", ""): v for k,v in weights.items()}
        weights = {k: v for k,v in weights.items() if k in model_dict and model_dict[k].shape == weights[k].shape}
        model_dict.update(weights)
        segmentor.load_state_dict(model_dict)
        logger.info("loaded checkpoint!")
        if len(weights) != len(model_dict):
            logger.warning("some weights were ignored due to mismatch")
            logger.warning(f"loaded {len(weights)}/{len(model_dict)} modules")
    else:
        logger.info("training from scratch")
#%%
config = HParams()
segmentor = Segmentor(config)
# %%
