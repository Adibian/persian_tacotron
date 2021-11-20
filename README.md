# persian_tacotron
Training Tacotron2 for Persian language as a Persian text-to-speech(TTS).
Tacotron2 is a TTS model that generates mel-spectrograms from text.
In this implementation we use [Tacotron2 from Nvidia](https://github.com/NVIDIA/tacotron2) and change it to train this model for persian language.
We clone Nvidia-Tacotron2 and install its requirements and then do following changes:
1. prepare persian data: many audio files and phonemes sequence for each file (we use phoneme instead of text because of using english characters and solving the problem of not writing some vowels in the Persian text)
2. change cleaner.py in tacotron2/text/ according to used characters in phonemes
3. change hparams.py in tacotron2/
4. create a python file that creates text file for model
5. create a python file that tests model for a phoneme


## How to use
To use this implementation:
1. clone this repository
2. install requirements in tacotron/requirments.txt 
3. add your data in files/: audio files to files/wavs and phoneme_transcriptions.txt to files/
4. run create_data_file.py to create text files for model in files/text_files 
5. move created files in files/text_files/ to tacotron/filelists/
6. change hparams.py in tacotron2/ to train model according to your data: epochs=? , iters_per_checkpoint=?, training_files='filelists/name-of-your-train-data.txt', validation_files='filelists/name-of-your-test-data.txt'
7. start training by following command:
    ```
    python tacotron2/train.py --output_directory=outdir --log_directory=logdir
    ```
    checkpoints will be saved in tacotron2/outdir/
    In training of model if you have 1000 audio files and batch-size is 16 so you will have 1000/16 iteration for any epochs.
    If you get an error about memory size, decrease batch_size in hparams.py to 8.
  
8. change get_results.py and set your test phonome in main: text = ?
9. run get_results.py and set parameter to last saved chackpoint file. for example to use 'checkpoint_32000' use following command:
    ```
    python get_results.py 32000
    ```
10. results of plot mel-spectrogram and audio file will be in results/


## results
After training model using 2500 audio files for about 400 epochs, results is this:
<img src="https://github.com/majidAdibian77/persian_tacotron/blob/master/result/plots/output_test2_56000(1634039244.8246922).jpg" width="1500"> 
And you can see some audio results [here](https://github.com/majidAdibian77/persian_tacotron/tree/master/result/wavs/).

