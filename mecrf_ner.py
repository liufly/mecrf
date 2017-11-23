from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from six.moves import range
from itertools import chain

import logging
import sys
import time

from mecrf_ner_cell import MemoryNetworkNERCell

def zero_nil_slot(t, name=None):
    """
    Overwrites the nil_slot (first row) of the input Tensor with zeros.

    The nil_slot is a dummy slot and should not be trained and influence
    the training algorithm.
    """
    with tf.name_scope(values=[t], name=name, default_name="zero_nil_slot") as name:
        t = tf.convert_to_tensor(t, name="t")
        s = tf.shape(t)[1]
        z = tf.zeros(tf.stack(values=[1, s]))
        return tf.concat(axis=0, values=[z, tf.slice(t, [1, 0], [-1, -1])], name=name)

class MECRF(object):
    """MECRF."""
    def __init__(
        self, 
        batch_size, 
        vocab_size, 
        answer_size,
        sentence_size,
        memory_size,
        embedding_size,
        rnn_hidden_size=200,
        mlp_hidden_size=64,
        max_grad_norm=5.0,
        initializer=tf.random_normal_initializer(stddev=0.1),
        optimizer=tf.train.AdamOptimizer(learning_rate=1e-2),
        session=None,
        name='MECRF',
        embedding_mat=None,
        update_embeddings=False,
        rnn_memory_hidden_size=200,
        nonlin=tf.nn.tanh,
        lexical_features_size=0,):

        self._batch_size = batch_size
        self._vocab_size = vocab_size
        self._answer_size = answer_size
        self._sentence_size = sentence_size
        self._memory_size = memory_size
        self._embedding_size = embedding_size
        self._max_grad_norm = max_grad_norm
        self._init = initializer
        self._opt = optimizer
        self._name = name
        self._rnn_hidden_size = rnn_hidden_size
        self._mlp_hidden_size = mlp_hidden_size
        
        self._embedding_mat = embedding_mat
        self._update_embeddings = update_embeddings
        self._rnn_memory_hidden_size = rnn_memory_hidden_size
        self._nonlin = nonlin
        self._lexical_features_size = lexical_features_size
        
        self._indices = tf.constant(
            np.arange(self._sentence_size).reshape(1, self._sentence_size, 1), 
            dtype=tf.float32
        ) # [1, sentence_size, 1]

        self._build_inputs()
        self._build_vars()
        
        # cross entropy
        sent_lens, unary_scores, log_likelihood, transition_params, link_logits = self._inference(
            self._memories,
            self._sentences,
            self._answers, 
            self._keep_prob,
            self._mem_idx,
            self._sent_lexical_features,
            self._mem_lexical_features,
        )
        # mem_lens: [None]
        # link_logits: [None, memory_size, memory_size + 1]
        
        # loss op
        nll = tf.negative(log_likelihood, name="negative_log_likelihood")
        loss_op = tf.reduce_mean(nll)
                    
        # gradient pipeline
        grads_and_vars = self._opt.compute_gradients(loss_op)
        total_mem_usage = 0
        grads_and_vars = filter(lambda x: x[0] is not None, grads_and_vars)
        grads_and_vars = [(tf.clip_by_norm(g, self._max_grad_norm), v) for g,v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self._nil_vars:
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        train_op = self._opt.apply_gradients(nil_grads_and_vars, name="train_op")

        # sentences_len
        self._sent_lens = sent_lens
        
        # unary_scores
        self._unary_scores_op = unary_scores
        
        # transition_params_op
        self._transition_params_op = transition_params
        
        # link predictions
        link_mask = tf.sequence_mask(
            lengths=tf.reshape(self._mem_idx + 1, shape=[-1]),
            maxlen=self._memory_size,
            dtype=tf.float32,
            name="link_mask_flattened"
        )
        link_mask = tf.reshape(
            link_mask, shape=[-1, self._sentence_size, self._memory_size]
        )
        self._link_predict_op = tf.argmax(
            input=link_mask * link_logits, dimension=2, name="link_predict_op"
        )
        self._link_predict_dist_op = link_mask * link_logits
        # [None, sentence_size, memory_size]
        
        # assign ops
        self.loss_op = loss_op
        self.train_op = train_op

        init_op = tf.global_variables_initializer()
        self._sess = session
        self._sess.run(init_op)

    def _build_inputs(self):
        self._memories = tf.placeholder(
            tf.int32, [None, self._memory_size], name="memories"
        )
        self._sentences = tf.placeholder(
            tf.int32, [None, self._sentence_size], name="sentences"
        )
        self._answers = tf.placeholder(
            tf.int32, [None, self._sentence_size], name="answers"
        )
        self._keep_prob = tf.placeholder(
            tf.float32, [], name="keep_prob"
        )
        self._mem_idx = tf.placeholder(
            tf.float32, [None, self._sentence_size], name="doc_start_index"
        )
        self._sent_lexical_features = tf.placeholder(
            tf.float32, [None, self._sentence_size, 
            self._lexical_features_size], 
            name="sentence_lexical_features"
        )
        self._mem_lexical_features = tf.placeholder(
            tf.float32, [None, self._memory_size, self._lexical_features_size], 
            name="memory_lexical_features"
        )

    def _build_vars(self):
        with tf.variable_scope(self._name):
            nil_word_slot = tf.zeros([1, self._embedding_size])
            
            EMB = None
            with tf.variable_scope("external_embedding") as emb_scope:
                self._emb = tf.get_variable(
                    name="EMB", 
                    shape=self._embedding_mat.shape, 
                    dtype=tf.float32, 
                    initializer=tf.constant_initializer(self._embedding_mat), 
                    trainable=False,
                )
            
            embedding_feature_size = self._embedding_size + self._lexical_features_size
            self._embedding_feature_size = embedding_feature_size
            hidden_size = embedding_feature_size

            self._hidden_size = hidden_size
            if self._rnn_memory_hidden_size == 0:
                self._rnn_memory_hidden_size = hidden_size

            self._rnn_memory_Ws_shape = [
                self._rnn_memory_hidden_size,
                self._embedding_feature_size,
            ]
            self._rnn_memory_bs_shape = [
                1,
                1,
                self._embedding_feature_size,
            ]

            hidden_output_size = self._embedding_feature_size
            
            self.RNN = tf.Variable(
                self._init([hidden_output_size, self._mlp_hidden_size]), 
                name="RNN"
            )
            
            self.RNN_b = tf.Variable(
                self._init([1, 1, self._mlp_hidden_size]), name="RNN_b"
            )
            
            self.RNN2TAG = tf.Variable(
                self._init([self._mlp_hidden_size, self._answer_size])
            )
            self.RNN2TAG_b = tf.Variable(self._init([1, 1, self._answer_size]))

        self._nil_vars = set([self._emb.name])
        
    def _tensor_dot(self, A, B):
        batch_size = tf.shape(A)[0]
        A_shape = A.get_shape().as_list()
        B_shape = B.get_shape().as_list()
        A_reshaped = tf.reshape(A, shape=[batch_size * A_shape[1], A_shape[2]])
        dot_prod = tf.matmul(A_reshaped, B)
        return tf.reshape(dot_prod, shape=[batch_size, A_shape[1], B_shape[1]])

    def _seq_len(self, seq):
        used = tf.sign(tf.abs(seq))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
    
    def _inference(self, memories, sentences, answers, keep_prob, mem_idx, 
                   sent_lexical_features, mem_lexical_features):
        with tf.variable_scope(self._name):
            memory_rnn_cell_fw = tf.contrib.rnn.GRUCell(
                self._rnn_memory_hidden_size
            )
            memory_rnn_cell_fw = tf.contrib.rnn.DropoutWrapper(
                memory_rnn_cell_fw, input_keep_prob=keep_prob, 
                output_keep_prob=keep_prob
            )
            memory_rnn_cell_bw = tf.contrib.rnn.GRUCell(
                self._rnn_memory_hidden_size
            )
            memory_rnn_cell_bw = tf.contrib.rnn.DropoutWrapper(
                memory_rnn_cell_bw, input_keep_prob=keep_prob, 
                output_keep_prob=keep_prob
            )

            mem_len = self._seq_len(memories)
            # [None]
            sent_len = self._seq_len(sentences)
            # [None]
            
            sent_emb = tf.nn.embedding_lookup(self._emb, sentences)
            # [None, sentence_size, emb_size]

            # m_emb = tf.nn.embedding_lookup(self._weight_matrices[0], memories)
            m_emb = tf.nn.embedding_lookup(self._emb, memories)
            # [None, memory_size, emb_size]
            c_emb = tf.nn.embedding_lookup(self._emb, memories)
            # [None, memory_size, emb_size]

            sent_emb = tf.concat(values=[sent_emb, sent_lexical_features], axis=2)
            # [None, sentence_size, emb_size + lexical_features_size]
            
            m_emb = tf.concat(values=[m_emb, mem_lexical_features], axis=2)
            # [None, memory_size, emb_size + lexical_features_size]
            c_emb = tf.concat(values=[c_emb, mem_lexical_features], axis=2)
            # [None, memory_size, emb_size + lexical_features_size]
        
            with tf.variable_scope("memory_rnn") as m_sentence_rnn_scope:
                (m_rnn_fw, m_rnn_bw), (_, _) = tf.nn.bidirectional_dynamic_rnn(
                    memory_rnn_cell_fw,
                    memory_rnn_cell_bw,
                    m_emb,
                    dtype=tf.float32,
                    sequence_length=mem_len,
                    scope=m_sentence_rnn_scope,
                    swap_memory=True,
                )
                # m_rnn_f/bw: [None, memory_size, rnn_memory_hidden_size]
                # m_rnn_state_f/bw: [None, rnn_memory_hidden_size]
            
                Wm_memory_rnn_fw = tf.get_variable(
                    initializer=self._init, 
                    shape=self._rnn_memory_Ws_shape, 
                    name="W_memory_rnn_fw",
                )
                Wm_memory_rnn_bw = tf.get_variable(
                    initializer=self._init,
                    shape=self._rnn_memory_Ws_shape,
                    name="W_memory_rnn_bw",
                )
                bm_memory_rnn = tf.get_variable(
                    initializer=self._init,
                    shape=self._rnn_memory_bs_shape,
                    name="b_memory_rnn"
                )
                m_rnn_output = self._nonlin(
                    self._tensor_dot(m_rnn_fw, Wm_memory_rnn_fw)
                        + self._tensor_dot(m_rnn_bw, Wm_memory_rnn_bw)
                        + bm_memory_rnn
                )
                # [None, memory_size, emb_size]
            
                m = m_rnn_output

                # sent_emb: [None, sentence_size, emb_size]
                W_sent_rnn_fw = tf.get_variable(
                    initializer=self._init, 
                    shape=self._rnn_memory_Ws_shape, 
                    name="W_sentence_rnn_fw",
                )
                W_sent_rnn_bw = tf.get_variable(
                    initializer=self._init,
                    shape=self._rnn_memory_Ws_shape,
                    name="W_sentence_rnn_bw",
                )
                b_sent_rnn = tf.get_variable(
                    initializer=self._init,
                    shape=self._rnn_memory_bs_shape,
                    name="b_sentence_rnn"
                )
                
                m_sentence_rnn_scope.reuse_variables()
                (sent_rnn_fw, sent_rnn_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    memory_rnn_cell_fw,
                    memory_rnn_cell_bw,
                    sent_emb,
                    dtype=tf.float32,
                    sequence_length=sent_len,
                    scope=m_sentence_rnn_scope,
                    swap_memory=True,
                )
                # sent_rnn_f/bw: [None, memory_size, rnn_memory_hidden_size]
                # sent_rnn_state_f/bw: [None, rnn_memory_hidden_size]
                sent_rnn_output = self._nonlin(
                    self._tensor_dot(sent_rnn_fw, W_sent_rnn_fw)
                        + self._tensor_dot(sent_rnn_bw, W_sent_rnn_bw)
                        + b_sent_rnn
                )
                # [None, memory_size, emb_size]
            sent_emb = sent_rnn_output

            mem_rnn_cell = MemoryNetworkNERCell(
                self._memory_size,
                self._embedding_feature_size,
                m,
                m,
                return_link=True,
            )
            
            mem_idx_expanded = tf.expand_dims(
                input=mem_idx, 
                axis=-1, 
                name="doc_start_index_reshaped"
            )

            (mem_rnn_output, mem_rnn_link), mem_rnn_state = tf.nn.dynamic_rnn(
                mem_rnn_cell,
                tf.tuple([sent_emb, mem_idx_expanded]),
                dtype=tf.float32,
                sequence_length=sent_len
            )
            # mem_rnn_output: [None, max_seq_len, hidden_size]
            # mem_rnn_link: [None, max_seq_len, max_seq_len]
            # mem_rnn_state: [None, hidden_size]
            
            rnn2mlp = self._tensor_dot(mem_rnn_output, self.RNN) + self.RNN_b
            # [None, sentence_size, mlp_hidden_size]
            mlp2tag = self._tensor_dot(rnn2mlp, self.RNN2TAG) + self.RNN2TAG_b
            # [None, sentence_size, answer_size]
            
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                mlp2tag, answers, sent_len)
            
            return sent_len, mlp2tag, log_likelihood, transition_params, mem_rnn_link

    def batch_fit(self, memories, sentences, answers, keep_prob, mem_idx, 
                  sent_lexical_features, mem_lexical_features):
        feed_dict = {
            self._memories: memories,
            self._sentences: sentences, 
            self._answers: answers, 
            self._keep_prob: keep_prob,
            self._mem_idx: mem_idx,
            self._sent_lexical_features: sent_lexical_features,
            self._mem_lexical_features: mem_lexical_features
        }
        loss, _ = self._sess.run(
            [self.loss_op, self.train_op], feed_dict=feed_dict
        )
        return loss
    
    def _get_mini_batch_start_end(self, n_train, batch_size=None):
        '''
        Args:
            n_train: int, number of training instances
            batch_size: int (or None if full batch)
        
        Returns:
            batches: list of tuples of (start, end) of each mini batch
        '''
        mini_batch_size = n_train if batch_size is None else batch_size
        batches = zip(
            range(0, n_train, mini_batch_size),
            list(range(mini_batch_size, n_train, mini_batch_size)) + [n_train]
        )
        return batches
    
    def predict(self, memories, sentences, mem_idx, sent_lexical_features, 
                mem_lexical_features):
        n_train = len(memories)
        batches = self._get_mini_batch_start_end(n_train, self._batch_size)
        unary_scores, transition_params, sentence_lens = [], None, []
        for start, end in batches:
            feed_dict = {
                self._memories: memories[start:end],
                self._sentences: sentences[start:end], 
                self._keep_prob: 1.0,
                self._mem_idx: mem_idx[start:end],
                self._sent_lexical_features: sent_lexical_features[start:end],
                self._mem_lexical_features: mem_lexical_features[start:end]
            }
            uss, transition_params, sls = self._sess.run(
                [self._unary_scores_op, self._transition_params_op, self._sent_lens],
                feed_dict=feed_dict,
            )
            unary_scores.extend(uss)
            sentence_lens.extend(sls)
            
        predictions = []
        for unary_score, seq_len in zip(unary_scores, sentence_lens):
            
            # Remove padding from the scores and tag sequence.
            us = unary_score[:seq_len]
            
            # Compute the highest scoring sequence.
            viterbi_sequence, _ = tf.contrib.crf.viterbi_decode(
                us, transition_params
            )
            predictions.append(viterbi_sequence)
            
        return predictions
    