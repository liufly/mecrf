from __future__ import absolute_import
from __future__ import print_function

import os, sys
import operator

from data_utils_ner import *
from mecrf_ner import *
from itertools import chain
from six.moves import range, reduce

import tensorflow as tf
import numpy as np
import cPickle as pickle

import sys
import logging
import uuid
import gc

from _collections import defaultdict

tf.flags.DEFINE_float("learning_rate", 0.01, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 5.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 10, "Evaluate and print results every x epochs")
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("epochs", 100, "Number of epochs to train for.")
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 500, "Maximum size of memory.")
tf.flags.DEFINE_integer("random_state", 101, "Random state.")
tf.flags.DEFINE_string("data_dir", "data/conll03-ner/", "Directory containing CoNLL-03-NER data")
tf.flags.DEFINE_integer("rnn_hidden_size", 20, "RNN hidden size [20]")
tf.flags.DEFINE_string("embedding_file", None, "Pre-trained word embedding file path [None]")
tf.flags.DEFINE_boolean("update_embeddings", False, "Update embeddings [False]")
tf.flags.DEFINE_boolean("bilinear", False, "Use bilinear [False]")
tf.flags.DEFINE_float("keep_prob", 1.0, "Keep prob [1.0]")
tf.flags.DEFINE_integer("mlp_hidden_size", 64, "MLP hidden state size [64]")
tf.flags.DEFINE_integer("rnn_memory_hidden_size", 0, "RNN memory hidden size [0]")
tf.flags.DEFINE_string("nonlin", "tanh", "Non-linearity [tanh]")

FLAGS = tf.flags.FLAGS

def get_ner_dict(data):
    ner2idx = {}
    for document in data:
        for sentence in document:
            for _, _, _, ner in sentence:
                if ner not in ner2idx:
                    ner2idx[ner] = len(ner2idx)
    return ner2idx

def load_embeddings(data, in_file, binary=False):
    # emb = word2vec.Word2Vec.load_word2vec_format(in_file, binary=binary)
    emb = {}
    unk = []
    with open(in_file) as in_f:
        nb_words, nb_dim = None, None
        for line in in_f:
            line = line.strip()
            attrs = line.split(' ')
            if len(attrs) == 2:
                nb_words = int(attrs[0])
                nb_dim = int(attrs[1])
                # self._embeddings = np.zeros((nb_words + 2, nb_dim), dtype=np.float32)
                continue
            word = attrs[0]
            word_emb = map(float, attrs[1:])
            emb[word] = word_emb
            unk.append(word_emb)
    # unk = np.mean(np.array(unk), axis=0)
    unk = emb['UNKNOWN']
    # print(len(emb))
    ret_emb = []
    ret_emb.append(np.zeros(len(unk))) # padding
    ret_emb.append(unk)
    ret_word2idx = {}
    for document in data:
        for sentence in document:
            for word, _, _, _ in sentence:
                if word.lower() in emb:
                    if word not in ret_word2idx:
                        ret_word2idx[word] = len(ret_emb)
                        ret_emb.append(emb[word.lower()])
                else:
                    ret_word2idx[word] = 1 # unk
    return np.asarray(ret_emb, dtype=np.float32), ret_word2idx

def output_conll(Gold, Pred, out_F):
    with open(out_F, 'w+') as f:
        assert len(Gold) == len(Pred)
        for gold, pred in zip(Gold, Pred):
            assert len(gold) == len(pred)
            for g, p in zip(gold, pred):
                f.write(' '.join([g[0], g[-1], p]))
                f.write('\n')
            f.write('\n')

regex_pattern = r'accuracy:\s+([\d]+\.[\d]+)%; precision:\s+([\d]+\.[\d]+)%; recall:\s+([\d]+\.[\d]+)%; FB1:\s+([\d]+\.[\d]+)'
def eval(gold, pred):
    out_filename = str(uuid.uuid4())
    cur_dir = os.path.dirname(__file__)
    out_abs_filepath = os.path.abspath(os.path.join(cur_dir, out_filename))
    try:
        output_conll(gold, pred, out_abs_filepath)
        cmd_process = os.popen(
            "perl " + os.path.abspath(os.path.join(cur_dir, "conlleval.pl")) + " < " + out_abs_filepath)
        cmd_ret = cmd_process.read()
        cmd_ret_str = str(cmd_ret)
        m = re.search(regex_pattern, cmd_ret)
        assert m is not None
        acc = float(m.group(1))
        precision = float(m.group(2))
        recall = float(m.group(3))
        f_score = float(m.group(4))
        return cmd_ret_str, acc, precision, recall, f_score
    except:
        return '', 0., 0., 0., 0.
    finally:
        # pass
        os.remove(out_abs_filepath)

if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    logger.info(" ".join(sys.argv))

    train = load_task(
        os.path.join(FLAGS.data_dir, 'eng.train'), 
        BIO=True, SBIEO=False,
    )
    train_flattened = [s for d in train for s in d]
    val = load_task(
        os.path.join(FLAGS.data_dir, 'eng.testa'), 
        BIO=True, SBIEO=False,
    )
    val_flattened = [s for d in val for s in d]
    test = load_task(
        os.path.join(FLAGS.data_dir, 'eng.testb'), 
        BIO=True, SBIEO=False,
    )
    test_flattened = [s for d in test for s in d]
    
    data = train + val + test
    data = np.asarray(data, dtype=np.object)
    
    assert FLAGS.embedding_file is not None
    embedding_mat, word2idx = load_embeddings(
        data, 
        FLAGS.embedding_file
    )
    idx2word = dict(zip(word2idx.values(), word2idx.keys()))
    FLAGS.embedding_size = embedding_mat.shape[1]
    
    logger.info('embedding_mat size: ' + str(embedding_mat.shape))

    np.random.seed(FLAGS.random_state)
    
    max_story_size = max([sum([len(s) for s in d]) for d in data])
    mean_story_size = int(np.mean([sum([len(s) for s in d]) for d in data]))
    sentence_size = max(map(len, chain.from_iterable(d for d in data)))
    
    memory_size = min(FLAGS.memory_size, max_story_size)
    
    ner2idx = get_ner_dict(data)
    idx2ner = dict(zip(ner2idx.values(), ner2idx.keys()))
    
    vocab_size = embedding_mat.shape[0]

    answer_size = len(ner2idx)
    
    logger.info("Longest sentence length %d" % sentence_size)
    logger.info("Longest story length %d" % max_story_size)
    logger.info("Average story length %d" % mean_story_size)
    
    # train/validation/test sets
    train_sentences, train_memories, train_answers, train_mem_idx = vectorize_data(train, word2idx, sentence_size, memory_size, ner2idx)
    val_sentences, val_memories, val_answers, val_mem_idx = vectorize_data(val, word2idx, sentence_size, memory_size, ner2idx)
    test_sentences, test_memories, test_answers, test_mem_idx = vectorize_data(test, word2idx, sentence_size, memory_size, ner2idx)
    
    train_sentence_lexical_features, train_memory_lexical_features = vectorize_lexical_features(train, sentence_size, memory_size)
    val_sentence_lexical_features, val_memory_lexical_features = vectorize_lexical_features(val, sentence_size, memory_size)
    test_sentence_lexical_features, test_memory_lexical_features = vectorize_lexical_features(test, sentence_size, memory_size)

    lexical_features_size = train_sentence_lexical_features.shape[2]

    logger.info("Training set title shape " + str(train_sentences.shape))
    logger.info("Training set text shape " + str(train_memories.shape))
    
    n_train = train_sentences.shape[0]
    n_test = test_sentences.shape[0]
    n_val = val_sentences.shape[0]
    
    logger.info("Training Size %d" % n_train)
    logger.info("Validation Size %d" % n_val)
    logger.info("Testing Size %d" % n_test)
    
    tf.set_random_seed(FLAGS.random_state)
    batch_size = FLAGS.batch_size
    
    global_step = None
    optimizer = None
    
    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate, epsilon=FLAGS.epsilon)
    
    batches = zip(range(0, n_train, batch_size), list(range(batch_size, n_train, batch_size)) + [n_train])
    batches = [(start, end) for start, end in batches]
    
    nonlin = None
    if FLAGS.nonlin == 'tanh':
        nonlin = tf.nn.tanh
    elif FLAGS.nonlin == 'relu':
        nonlin = tf.nn.relu
    else:
        raise
    
    best_val = -1
    best_val_perf, best_test_perf = None, None

    session_conf = tf.ConfigProto(
        intra_op_parallelism_threads=4,
        inter_op_parallelism_threads=4
    )

    with tf.Session(config=session_conf) as sess:
        tf.set_random_seed(seed=FLAGS.random_state)
        model = MECRF(
            batch_size,
            vocab_size,
            answer_size,
            sentence_size,
            memory_size,
            FLAGS.embedding_size,
            session=sess,
            max_grad_norm=FLAGS.max_grad_norm,
            optimizer=optimizer,
            embedding_mat=embedding_mat,
            rnn_hidden_size=FLAGS.rnn_hidden_size,
            mlp_hidden_size=FLAGS.mlp_hidden_size,
            rnn_memory_hidden_size=FLAGS.rnn_memory_hidden_size,
            nonlin=nonlin,
            lexical_features_size=lexical_features_size,
        )
        
        for t in range(1, FLAGS.epochs+1):
            np.random.shuffle(batches)
            total_cost = 0.0
            
            for start, end in batches:
                m = train_memories[start:end]
                s = train_sentences[start:end]
                a = train_answers[start:end]
                mi = train_mem_idx[start:end]
                slf = train_sentence_lexical_features[start:end]
                mlf = train_memory_lexical_features[start:end]
                cost_t = model.batch_fit(
                    m, s, a, FLAGS.keep_prob, mi, slf, mlf
                )
                total_cost += cost_t
                
            if t % FLAGS.evaluation_interval == 0:
                train_preds = []
                for start in range(0, n_train, batch_size):
                    end = start + batch_size
                    m = train_memories[start:end]
                    s = train_sentences[start:end]
                    a = train_answers[start:end]
                    # temporal = get_temporal_encoding(m, random_time=0.0)
                    mi = train_mem_idx[start:end]
                    slf = train_sentence_lexical_features[start:end]
                    mlf = train_memory_lexical_features[start:end]
                    pred = model.predict(m, s, mi, slf, mlf)
                    train_preds += list(pred)
    
                train_scores, acc, precision, recall, f_score = eval(
                    train_flattened,
                    [[idx2ner[p] for p in pred] for pred in train_preds]
                )
                
                logger.info('-----------------------')
                logger.info('Epoch %d' % t)
                logger.info('Total Cost: %f' % total_cost)
                logging.info('Training acc: %.2f, precision: %.2f, recall: %.2f, f_score: %.2f' % (acc, precision, recall, f_score))
                logging.info('Training: ' + train_scores)
    
                val_preds = model.predict(val_memories, val_sentences, val_mem_idx, val_sentence_lexical_features, val_memory_lexical_features)
                val_scores, acc, precision, recall, f_score = eval(
                    val_flattened,
                    [[idx2ner[p] for p in pred] for pred in val_preds]
                )
                logging.info('Validation acc: %.2f, precision: %.2f, recall: %.2f, f_score: %.2f' % (acc, precision, recall, f_score))
                logging.info('Validation: ' + val_scores)
                
                val_f_score = f_score
                val_perf = (acc, precision, recall, f_score)

                test_preds = model.predict(test_memories, test_sentences, test_mem_idx, test_sentence_lexical_features, test_memory_lexical_features)
                test_scores, acc, precision, recall, f_score = eval(
                    test_flattened,
                    [[idx2ner[p] for p in pred] for pred in test_preds]
                )
                logging.info('Test acc: %.2f, precision: %.2f, recall: %.2f, f_score: %.2f' % (acc, precision, recall, f_score))
                logging.info('Testing: ' + test_scores)

                test_perf = (acc, precision, recall, f_score)

                if val_f_score > best_val:
                    best_val = val_f_score
                    best_val_perf = val_perf
                    best_test_perf = test_perf

                logger.info('-----------------------')

        logger.info('Best validation acc: %.2f, precision: %.2f, recall: %.2f, f_score: %.2f' % (best_val_perf))
        logger.info('Best test acc: %.2f, precision: %.2f, recall: %.2f, f_score: %.2f' % (best_test_perf))
            

    
