import argparse
import tensorflow as tf
from EDL.utils import misc_utils as utils
import EDL.model as train 
import os

def add_arguments(parser):
    """Build ArgumentParser."""
    parser.add_argument("--src", type=str, default='in')
    parser.add_argument("--tgt", type=str, default='out')
    parser.add_argument("--train_prefix", type=str, default='../data/entity_dection/train')
    parser.add_argument("--dev_prefix", type=str, default='../data/entity_dection/dev')
    parser.add_argument("--test_prefix", type=str, default='../data/entity_dection/test')
    parser.add_argument("--vocab", type=str, default='../data/entity_dection/vocab.in')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--infer_size", type=int, default=32)
    parser.add_argument("--num_layer", type=int, default=1)
    parser.add_argument("--num_units", type=int, default=300)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--forget_bias", type=float, default=1.0,
                      help="Forget bias for BasicLSTMCell.")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--optimizer", type=str, default="sgd", help="sgd | adam")
    parser.add_argument("--learning_rate", type=float, default=1.0,
                      help="Learning rate. Adam: 0.001 | 0.0001")
    parser.add_argument("--num_train_steps", type=int, default=500000)
    parser.add_argument("--num_display_steps", type=int, default=1000)
    parser.add_argument("--num_eval_steps", type=int, default=10000)

    
def create_hparams(flags):
    """Create ArgumentParser."""
    return tf.contrib.training.HParams(
        src=flags.src,
        tgt=flags.tgt,
        train_prefix=flags.train_prefix,
        dev_prefix=flags.dev_prefix,
        test_prefix=flags.test_prefix,
        vocab=flags.vocab,
        batch_size=flags.batch_size,
        num_layer=flags.num_layer,
        num_units=flags.num_units,
        dropout=flags.dropout,
        forget_bias=flags.forget_bias,
        hidden_size=flags.hidden_size,
        optimizer=flags.optimizer,
        learning_rate=flags.learning_rate,
        num_train_steps=flags.num_train_steps,
        num_display_steps=flags.num_display_steps,
        num_eval_steps=flags.num_eval_steps,
        infer_size=flags.infer_size,
        )


def extend_hparams(hparams):
    """Build files paths"""
    hparams.train_src=hparams.train_prefix+'.'+hparams.src
    hparams.train_tgt=hparams.train_prefix+'.'+hparams.tgt
    hparams.dev_src=hparams.dev_prefix+'.'+hparams.src
    hparams.dev_tgt=hparams.dev_prefix+'.'+hparams.tgt    
    hparams.test_src=hparams.test_prefix+'.'+hparams.src
    hparams.test_tgt=hparams.test_prefix+'.'+hparams.tgt
    
def infer(path):
    """Build inference model for entity detection
    Args:
        path: the path to the model
    Returns:
        model: the detection model
    """
    hparams=utils.load_hparams(path)
    infer_sess,infer_model=train.infer(hparams,path)
    return infer_sess,infer_model

if __name__ == "__main__":
  nmt_parser = argparse.ArgumentParser()
  add_arguments(nmt_parser)
  FLAGS, unparsed = nmt_parser.parse_known_args()
  hparams=create_hparams(FLAGS)
  extend_hparams(hparams)  
  utils.print_hparams(hparams)
  train.train(hparams)