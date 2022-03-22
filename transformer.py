import tensorflow as tf

from data_load import load_vocab
from modules import feed_forward, multihead_attention, positional_encoding, get_token_embeddings

class Transformer:
    def __init__(self, param):
        self.params = param
        self.token_idx, self.idx_token = load_vocab(param.vocab)
        self.embedding = get_token_embeddings(self.params.vocab_size, self.params.d_model, zero_pad=True)

    '''
    '''
    def encoder(self, sent_src, training=True):
        with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
            x, _, src_sentence = sent_src
            src_mask = tf.math.equal(x, 0)

            # embedding
            encoder = tf.nn.embedding_lookup(self.embedding, x)
            encoder += positional_encoding(encoder, self.params.max_len_src)
            # encoder = tf.layers.dropout(encoder, self.params.dropout_rate, training=training)

            # encoder block
            for i in range(self.params.num_blocks):
                with tf.variable_scope("num_block {}".format(i), reuse=tf.AUTO_REUSE):
                    encoder = multihead_attention(q=encoder, k=encoder, v=encoder, 
                                                    num_head=self.params.num_att_head)
                    encoder = feed_forward(encoder, num_node=[self.params.d_net, self.params.d_model])

        context = encoder
        
        return context, src_sentence, src_mask

    '''
    '''
    def decoder(self, context, src_mask, sent_tgt, training=True):
        with tf.variable_scope('decoder', reuse=tf.REUSE):
            decoder_input, y, _,tgt_sentence = sent_tgt
            tgt_mask = tf.math.equal(decoder_input, 0)
            
            # embedding
            decoder = tf.nn.embedding_lookup(self.embedding, decoder_input)
            decoder += positional_encoding(decoder, self.params.max_len_tgt)

            # decoder block


        weights = tf.transpose(self.embedding)
        logits = tf.einsum('ntd,dk->ntk', decoder, weights)
        y_hat = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits, y_hat, y, tgt_sentence

    '''
    '''
    def train(self, sent_src, sent_tgt):

        return 0

