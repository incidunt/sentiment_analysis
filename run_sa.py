# -*- coding: utf-8 -*-
"""
Created on Thur Mar 2 2017

@author: Aiting Liu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time

import numpy as np
import tensorflow as tf

import data_utils
import sa_model

import subprocess

tf.app.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")
# tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.9,
#                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 128,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 128, "Size of each model layer.")
tf.app.flags.DEFINE_integer("word_embedding_size", 100, "Size of the word embedding")
tf.app.flags.DEFINE_integer("num_layers", 1, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("sent_vocab_size", 110000, "max vocab Size.")
# tf.app.flags.DEFINE_integer("out_vocab_size", 500, "max tag vocab Size.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./log", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("max_training_steps", 1000000,
                            "Max training steps.")
tf.app.flags.DEFINE_integer("max_test_data_size", 0,
                            "Max size of test set.")
# tf.app.flags.DEFINE_boolean("use_attention", False,
#                             "Use attention based RNN")
tf.app.flags.DEFINE_integer("max_sequence_length", 250,
                            "Max sequence length.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.8,
                          "dropout keep cell input and output prob.")
# tf.app.flags.DEFINE_boolean("bidirectional_rnn", False,
#                             "Use birectional RNN")
# tf.app.flags.DEFINE_string("task", 'joint', "Options: joint; label; tagging")
FLAGS = tf.app.flags.FLAGS

if FLAGS.max_sequence_length == 0:
    print('Please indicate max sequence length. Exit')
    exit()


def create_model(session, sent_vocab_size, label_vocab_size):
    """Create model and initialize or load parameters in session."""
    with tf.variable_scope("model", reuse=None):
        model_train = sa_model.SaModel(
            sent_vocab_size, label_vocab_size, FLAGS.max_sequence_length,
            FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
            learning_rate=FLAGS.learning_rate,
            dropout_keep_prob=FLAGS.dropout_keep_prob, use_lstm=True,
            forward_only=False)
    with tf.variable_scope("model", reuse=True):
        model_test = sa_model.SaModel(
            sent_vocab_size, label_vocab_size, FLAGS.max_sequence_length,
            FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
            learning_rate=FLAGS.learning_rate,
            dropout_keep_prob=FLAGS.dropout_keep_prob, use_lstm=True,
            forward_only=True)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model_train.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model_train, model_test


def train():
    print('Applying Parameters:')
    for k, v in FLAGS.__dict__['__flags'].items():
        print('%s: %s' % (k, str(v)))
    print("Preparing data in %s" % FLAGS.data_dir)
    sent_train, label_train, \
    sent_valid, label_valid, \
    sent_test, label_test, \
    sent_vocab_path, label_vocab_path = data_utils.prepare_multi_task_data(
        FLAGS.data_dir, FLAGS.sent_vocab_size)

    result_dir = FLAGS.data_dir + '/test_results'
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    current_valid_out_file = result_dir + '/valid_hyp'
    current_test_out_file = result_dir + '/test_hyp'

    sent_vocab, rev_sent_vocab = data_utils.initialize_vocabulary(sent_vocab_path)
    label_vocab, rev_label_vocab = data_utils.initialize_vocabulary(label_vocab_path)
    print(rev_label_vocab)

    sent_vocab_size = len(sent_vocab)
    label_vocab_size = len(label_vocab)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Create model.
        print("Max sequence length: %d." % FLAGS.max_sequence_length)
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

        sess.run(tf.global_variables_initializer())

        model, model_test = create_model(sess, sent_vocab_size, label_vocab_size)
        print("Creating model with sent_vocab_size=%d,"
              "and label_vocab_size=%d." % (sent_vocab_size, label_vocab_size))

        # Read data into buckets and compute their sizes.
        print("Reading train/valid/test data (training set limit: %d)."
              % FLAGS.max_train_data_size)
        valid_set = data_utils.read_data(sent_valid, label_valid)
        test_set = data_utils.read_data(sent_test, label_test)
        train_set = data_utils.read_data(sent_train, label_train)

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0

        best_valid_score = 0
        best_test_score = 0

        while model.global_step.eval() < FLAGS.max_training_steps:
            # Get a batch and make a step.
            start_time = time.time()

            batch_inputs, batch_labels, batch_sequence_length = model.get_batch(train_set)
            # print(batch_inputs[0].shape)

            _, step_loss, logits = model.step(sess, batch_inputs, batch_labels, batch_sequence_length, False)

            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print("global step %d step-time %.2f. Training perplexity %.2f"
                      % (model.global_step.eval(), step_time, perplexity))
                sys.stdout.flush()
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0

                def write_eval_result(result_list, result_path):
                    with tf.gfile.GFile(result_path, 'w') as f:
                        for i in range(len(result_list)):
                            f.write(result_list[i] + '\n')

                def run_valid_test(data_set, mode):  # mode: Eval, Test
                    # Run evals on development/test set and print the accuracy.
                    ref_label_list = list()
                    hyp_label_list = list()
                    label_correct_count = 0

                    # accuracy = 0.0

                    eval_loss = 0.0
                    count = 0
                    for i in range(len(data_set)):
                        count += 1
                        inputs, labels, sequence_length = model_test.get_one(data_set, i)

                        _, _step_loss, logits = model_test.step(sess, inputs, labels, sequence_length, True)
                        eval_loss += _step_loss / len(data_set)

                        ref_label = np.argmax(labels)
                        ref_label_list.append(rev_label_vocab[ref_label])
                        hyp_label = np.argmax(logits[0])
                        hyp_label_list.append(rev_label_vocab[hyp_label])

                        if ref_label == hyp_label:
                            label_correct_count += 1

                    label_accuracy = float(label_correct_count) * 100 / count

                    print("  %s label_accuracy: %.2f %d/%d" % (mode, label_accuracy, label_correct_count, count))
                    sys.stdout.flush()
                    out_file = None
                    if mode == 'Valid':
                        out_file = current_valid_out_file
                    elif mode == 'Test':
                        out_file = current_test_out_file

                    write_eval_result(hyp_label_list, out_file)  # write prediction result to output file path

                    return label_accuracy, hyp_label_list

                # valid
                valid_label_accuracy, hyp_list = run_valid_test(valid_set, 'Valid')
                if valid_label_accuracy > best_valid_score:
                    best_valid_score = valid_label_accuracy
                    # save the best output file
                    subprocess.call(['mv', current_valid_out_file,
                                     current_valid_out_file + '_best_acc_%.2f' % best_valid_score])
                # test, run test after each validation for development purpose.
                test_label_accuracy, hyp_list = run_valid_test(test_set, 'Test')
                if test_label_accuracy > best_test_score:
                    best_test_score = test_label_accuracy
                    # save the best output file
                    subprocess.call(['mv', current_test_out_file,
                                     current_test_out_file + '_best_acc_%.2f' % best_test_score])


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
