"""
Usage: main.py [options] --dataroot <dataroot> --cuda
"""

import os
import argparse
import torch.backends.cudnn as cudnn

from train import Trainer

from preprocess import Preprocess

def main(config):
    os.system('mkdir {0}'.format(config.outf))


    cudnn.benchmark = True

    # dataroot, cache, image_size, n_channels, image_batch, video_batch, video_length):
    data_loader = Preprocess(config.dataroot, config.window_size)
    print("Data preprocessing Finished")

    trainer = Trainer(config, data_loader)

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataroot",default="C:\\Users\msi\Desktop\Soohyun\CHALLENGERS\DATASET\\Korean_bad_words.txt")
    parser.add_argument("--window_size", default=1)
    parser.add_argument("--embedding_dim",default=128, type=int)
    parser.add_argument("--outf", default="outf")
    parser.add_argument("--cuda", type=int, default=1)
    parser.add_argument("--model_type", type=str, default="NS",help="CBOW | Skip-Gram | NS (Negative Sampling)")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr",type=float, default=0.1)
    config = parser.parse_args()

    main(config)
