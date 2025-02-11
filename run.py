# %%- imports
from model import Segmentor
from loguru import logger
import torch
import torchaudio
import torchaudio.transforms as T
import IPython.display as ipd
# %% - define hparams
class HParams:
    wav_path = ''
    dataset = 'timit'  # choices=['timit', 'buckeye']
    # 'directory for saving run outputs (logs, ckpt, etc.)'
    run_dir = '/tmp/segmentation'
    exp_name = 'segmentation_experiment'  # help='experiment name')
    # 'path to a pre-trained model, if provided, training will resume from that point')
    load_ckpt = 'segmentor.ckpt'
    gpus = '-1'  # help='gpu ids to use, -1 for cpu')
    devrun = False  # action='store_true', help='dev run on a dataset of size `devrun_size`')
    devrun_size = 10  # help='size of dataset for dev run')
    # action='store_true', help='flag to indicate to run a test epoch only (not training will take place)')
    test = False

    lr = 0.001  # help='initial learning rate')
    optimizer = 'adam'
    momentum = 0.9
    epochs = 50  # help='upper epoch limit')
    batch_size = 1  # metavar='N', help='batch size')
    dropout = 0.0  # help='dropout probability value')
    seed = 1245  # help='random seed')
    patience = 5  # help='patience for early stopping')
    gamma = 0.15  # help='gamma margin')
    overfit = -1  # help='gamma margin')
    val_percent_check = 1.0  # help='how much of the validation set to check')
    val_check_interval = 1.0  # help='validation check every K epochs')
    val_ratio = 0.1  # help='precentage of validation from train')

    # rnn_input_size = 1  # help='number of inputs')
    rnn_dropout = 0.3  # help='dropout')
    rnn_hidden_size = 200  # help='RNN hidden layer size')
    birnn = True  # help='BILSTM, if define will be biLSTM')
    rnn_layers = 2  # help='number of lstm layers')
    # help='minimal size of segment, examples with segments smaller than this will be ignored')
    min_seg_size = 1
    max_seg_size = 100  # help='see `min_seg_size`')
    max_len = 500  # help='maximal size of sequences')
    # choices=["mfcc", "mel", "spect"], help='type of acoustic features to use')
    feats = "mel"
    # action="store_true", help='if this flag is on seuqences will be randomly trimmed')
    random_trim = False
    # action="store_true", help='if this flag is on delta features will be added')
    delta_feats = False
    # action="store_true", help='if this flag is on the euclidean features will be added (see paper)')
    dist_feats = False
    normalize = False  # action="store_true", help='flag to normalize features')
    bin_cls = 0  # type=float, help='coefficient of binary classification loss')
    phn_cls = 0  # type=float, help='coefficient of phoneme classification loss')
    n_fft = 160  # help='n_fft for feature extraction')
    hop_length = 160  # help='hop_length for feature extraction')
    n_mels = 40  # help='number of mels')
    n_mfcc = 13  # help='number of mfccs')

    cuda = torch.cuda.is_available()

    n_classes = {'timit': 39, 'buckeye': 40}[dataset]
    rnn_input_size = {'mfcc':  n_mfcc * (3 if delta_feats else 1) + (4 if dist_feats else 0),
                            'mel':   n_mels,
                            'spect': n_fft / 2 + 1}[feats]

# %%


def build_model(config):
    config = HParams()
    segmentor = Segmentor(config)

    if config.load_ckpt not in [None, "None"]:
        logger.info(f"loading checkpoint from: {config.load_ckpt}")
        model_dict = segmentor.state_dict()
        # weights = torch.load(config.load_ckpt, map_location='cuda:0')["state_dict"] gpu load
        weights = torch.load(config.load_ckpt, map_location='cpu')[
            "state_dict"]
        weights = {k.replace("segmentor.", ""): v for k, v in weights.items()}
        weights = {k: v for k, v in weights.items(
        ) if k in model_dict and model_dict[k].shape == weights[k].shape}
        model_dict.update(weights)
        segmentor.load_state_dict(model_dict)
        logger.info("loaded checkpoint!")
        if len(weights) != len(model_dict):
            logger.warning("some weights were ignored due to mismatch")
            logger.warning(f"loaded {len(weights)}/{len(model_dict)} modules")
    else:
        logger.info("training from scratch")
    return segmentor


# %%
config = HParams()
# %%
file_path = 'libri-audio/1272-128104-0011.flac'
speech, sr = torchaudio.load(file_path)

# %%
segmentor = build_model(config)
# %% - transform
if config.feats == "mfcc":
    print('mfcc')
    transform = T.MFCC(
        sample_rate=sr,
        n_mfcc=config.n_mfcc,
        melkwargs={"n_fft": config.n_fft, "hop_length": config.hop_length, "n_mels": config.n_mels}
    )
elif config.feats == "mel":
    print('mel')
    transform = T.MelSpectrogram(
        sample_rate=sr,
        n_mels=config.n_mels,
        n_fft=config.n_fft,
        win_length=config.n_fft,
        hop_length=config.hop_length
    )
else:
    raise ValueError(f"unknown feature type: {config.feats}")

# Convert waveform
transformed_spec = transform(speech)

# Ensure correct shape: (batch_size=1, time_steps, input_size=n_mfcc)
transformed_spec = transformed_spec.squeeze(0).transpose(0, 1).unsqueeze(0)  # Shape: (1, time_steps, 13)

length = torch.tensor([transformed_spec.shape[1]])  # Sequence length

# %%
results = segmentor(transformed_spec, length)
# %%
results['pred']
# %%
# Convert MFCC frame indices to sample indices
segmets = results['pred'][0]
sample_segments = [frame * config.hop_length for frame in segmets]
print(sample_segments, config.hop_length)
# Split the audio using the converted sample indices
audio_segments = [speech[:,start:end] for start, end in zip(sample_segments, sample_segments[1:])]
# print(len(speech))
# ipd.display(ipd.Audio(speech[:,0:160].numpy(), rate=sr))
# Save each segment as a separate WAV file
for i, segment in enumerate(audio_segments):
    ipd.display(ipd.Audio(segment.numpy(), rate=sr))


# %%
