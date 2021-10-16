import matplotlib
import matplotlib.pylab as plt
import IPython.display as ipd
import numpy as np
import torch
import time
import pickle


def plot_data(data, file_name, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='lower', 
                       interpolation='none')
    plt.savefig("result/plots/output_{0}({1}).jpg".format(file_name, time.time()))
    

def load_trained_model(checkpoint_path):
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    _ = model.cuda().eval().half()
    return model
   
def load_vocoder():
    waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16', force_download=True)
    waveglow = waveglow.remove_weightnorm(waveglow)
    waveglow = waveglow.to('cuda')
    waveglow.eval().half()
    for k in waveglow.convinv:
        k.float()
    return waveglow

def prepaire_input(text):
    t2s = text_to_sequence(text, ['english_cleaners'])
    sequence = np.array(t2s)[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()
    return sequence

def get_result_audio(mel_outputs_postnet, denoiser, waveglow, file_name):
    from scipy.io.wavfile import write
    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)
    write("result/wavs/audio_{0}({1}).wav".format(file_name, time.time()), rate, audio_numpy)


def save_mel_object(mel, file_name):
    pickle_out = open("result/mel_object/mel_" + file_name + ".pkl","wb")
    pickle.dump(mel, pickle_out)
    pickle_out.close()


if __name__ == "__main__":
    # waveglow = load_vocoder()
    
    import sys
    sys.path.append('tacotron2/')
    from hparams import create_hparams
    from model import Tacotron2
    from layers import TacotronSTFT, STFT
    from audio_processing import griffin_lim
    from train import load_model
    from text import text_to_sequence
    from waveglow.denoiser import Denoiser

    # denoiser = Denoiser(waveglow).half()
    # itter = "32000"
    itter = sys.argv[1]
    model = load_trained_model('tacotron2/outdir/checkpoint_' + itter)
    text = 'BEHHAMINDALILiYEKIhAZBEHTARINDaRUHaYEGIYaHIBARaYEDARMaNiLARZEsEDASTEihASABIMIBasADi'
    sequence = prepaire_input(text)
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    plot_data((mel_outputs.float().data.cpu().numpy()[0],
               mel_outputs_postnet.float().data.cpu().numpy()[0],
               alignments.float().data.cpu().numpy()[0].T), "test2_"+itter)
    # get_result_audio(mel_outputs_postnet, denoiser, waveglow, "test1_"+itter)
    
    save_mel_object(mel_outputs_postnet, "test2_"+itter)
