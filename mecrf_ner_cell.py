'''
Created on 15Dec.,2016

@author: fei
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh

import numpy as np
import tensorflow as tf
import tensorflow

from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib.rnn import RNNCell

class MemoryNetworkNERCell(RNNCell):
    
    def __init__(self, max_seq_len, emb_size, M, C, return_link=True):

        self._max_seq_len = max_seq_len
        self._emb_size = emb_size
        self._padding = tf.zeros(shape=[1, 1, self._emb_size], dtype=tf.float32, name='padding')
        self._return_link = return_link
        self._M = M # [None, max_seq_len, hidden_size]
        self._C = C # [None, max_seq_len, hidden_size]
        
    @property
    def state_size(self):
        return self._emb_size

    @property
    def output_size(self):
        if self._return_link:
            return (LSTMStateTuple(self._emb_size, self._max_seq_len))
        else:
            return self._emb_size
    
    def _memory_length(self, memory):
        '''
        memory: (None, self._memory_size, self._embedding_size)
        '''
        used = tf.sign(tf.reduce_max(tf.abs(memory), reduction_indices=2))
        length = tf.reduce_sum(used, reduction_indices=1)
        length = tf.cast(length, tf.int32)
        return length
    
    def _construct_current_memory(self, mem, idx):
        '''
        mem: [None, max_seq_len, emb_size]
        idx: [None]
        '''
        
        mask = tf.sequence_mask(lengths=idx, maxlen=self._max_seq_len, dtype=tf.float32)
        # [None, max_seq_len]
        mask = tf.expand_dims(input=mask, axis=-1)
        # [None, max_seq_len, 1]
        
        assert_op1 = tf.Assert(tf.equal(self._max_seq_len, tf.shape(mem)[1]), [mem])
        assert_op2 = tf.Assert(tf.equal(self._max_seq_len, tf.shape(mask)[1]), [mask])
        with tf.control_dependencies([assert_op1, assert_op2]):
            m = mem * mask
            # [None, max_seq_len, emb_size]
            return m
        
    def _softmax_with_mask(self, u, mask):
        '''
        u: [None, memory_size]
        mask: [None, memory_size]
        '''
        u_tmp = u - tf.reduce_max(u, 1, keep_dims=True) 
        # [None, memory_size]
        exp_u_tmp = tf.exp(u_tmp) 
        # [None, memory_size]
        masked_exp = exp_u_tmp * mask 
        # [None, memory_size]
        sum_2d = tf.expand_dims(input=tf.reduce_sum(masked_exp, 1), axis=-1) 
        # [None, 1]
        p = tf.div(masked_exp, sum_2d, name='p') 
        # [None, memory_size]
        return p
        
    
    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or type(self).__name__):  
            # "MemoryNetworkCell"
            u, i = inputs
            i = tf.cast(tf.reduce_sum(i, 1), dtype=tf.int32)
            # u: [None, emb_size]
            # i: [None], dtype=tf.int32

            m = self._M 
            # [None, memory_size, emb_size]
            c = self._C 
            # [None, memory_size, emb_size]
            
            u_temp = tf.expand_dims(input=u, axis=1) 
            # [None, 1, emb_size]
            dotted = tf.reduce_sum(m * u_temp, 2) 
            # [None, memory_size]

            # Calculate probabilities
            mem_mask = tf.sequence_mask(lengths=i + 1, maxlen=self._max_seq_len, dtype=tf.float32)
            probs = self._softmax_with_mask(dotted, mem_mask) 
            # [None, memory_size]
            probs_temp = tf.expand_dims(input=probs, axis=1) 
            # [None, 1, memory_size]
            o_k = tf.reduce_sum(tf.matmul(probs_temp, c), 1)
            # o_k: [None, emb_size]
            
            u_out = u + o_k
            
            if self._return_link:
                return LSTMStateTuple(u_out, dotted), u_out
            else:
                return u_out, u_out
          
        
