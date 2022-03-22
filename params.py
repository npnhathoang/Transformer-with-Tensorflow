import argparse
from ast import arg

class params:
    parser = argparse.ArgumentParser()

    # prepro
    parser.add_argument('--vocab_size', default=20000, type=int)

    # training
    parser.add_argument('--batch_size', default=120, type=int)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.004, type=float)
    parser.add_argument('--warmup_steps', default=2000, type=int)

    # model config
    parser.add_argument('--d_model', default=512, type=int)
    parser.add_argument('--num_blocks', default=6, type=int)
    parser.add_argument('--num_att_heads', default=8, type=int)
    parser.add_argument('--d_net', default=2048, type=int)
    parser.add_argument('--max_len_src', default=100, type=int)
    parser.add_argument('--max_len_tgt', default=100, type=int)
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    