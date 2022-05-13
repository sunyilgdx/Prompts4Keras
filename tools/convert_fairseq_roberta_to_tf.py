import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import torch
import numpy as np
import tensorflow as tf
from transformers import BertModel
from fairseq.models.roberta import RobertaModel as FairseqRobertaModel


class BertConfig:
    def __init__(self,
                 vocab_size_or_config_json_file=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 **kwargs):
        super(BertConfig, self).__init__(**kwargs)
        self.vocab_size = vocab_size_or_config_json_file
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps


def convert_roberta_checkpoint_to_tf(roberta_checkpoint_path, ckpt_dir, model_name):
    """
    Copy/paste/tweak roberta's weights to our BERT structure.
    """
    roberta = FairseqRobertaModel.from_pretrained(roberta_checkpoint_path)
    roberta.eval()  # disable dropout
    config = BertConfig(
        vocab_size_or_config_json_file=50265,
        hidden_size=roberta.args.encoder_embed_dim,
        num_hidden_layers=roberta.args.encoder_layers,
        num_attention_heads=roberta.args.encoder_attention_heads,
        intermediate_size=roberta.args.encoder_ffn_embed_dim,
        max_position_embeddings=514,
        type_vocab_size=1,
        layer_norm_eps=1e-5,  # PyTorch default used in fairseq
    )
    print("Our BERT config:", config)

    tensors_to_transpose = (
        "dense.weight",
        "self_attn.k_proj.weight",
        "self_attn.q_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.out_proj.weight",
        "fc1.weight",
        "fc2.weight"
    )

    var_map = (
        ('sentence_encoder.embed_tokens.weight', 'bert/embeddings/word_embeddings'),
        ('sentence_encoder.embed_positions.weight', 'bert/embeddings/position_embeddings'),
        ('sentence_encoder.emb_layer_norm.weight', 'bert/embeddings/LayerNorm/gamma'),
        ('sentence_encoder.emb_layer_norm.bias', 'bert/embeddings/LayerNorm/beta'),
        ('lm_head.dense.weight', 'cls/predictions/transform/dense/kernel'),
        ('lm_head.dense.bias', 'cls/predictions/transform/dense/bias'),
        ('lm_head.layer_norm.weight', 'cls/predictions/transform/LayerNorm/gamma'),
        ('lm_head.layer_norm.bias', 'cls/predictions/transform/LayerNorm/beta'),
        ('lm_head.bias', 'cls/predictions/output_bias'),
        ('sentence_encoder.layers.', 'bert/encoder/layer_'),
        ('self_attn.q_proj.weight', 'attention/self/query/kernel'),
        ('self_attn.q_proj.bias', 'attention/self/query/bias'),
        ('self_attn.k_proj.weight', 'attention/self/key/kernel'),
        ('self_attn.k_proj.bias', 'attention/self/key/bias'),
        ('self_attn.v_proj.weight', 'attention/self/value/kernel'),
        ('self_attn.v_proj.bias', 'attention/self/value/bias'),
        ('self_attn.out_proj.weight', 'attention/output/dense/kernel'),
        ('self_attn.out_proj.bias', 'attention/output/dense/bias'),
        ('self_attn_layer_norm.weight', 'attention/output/LayerNorm/gamma'),
        ('self_attn_layer_norm.bias', 'attention/output/LayerNorm/beta'),
        ('fc1.weight', 'intermediate/dense/kernel'),
        ('fc1.bias', 'intermediate/dense/bias'),
        ('fc2.weight', 'output/dense/kernel'),
        ('fc2.bias', 'output/dense/bias'),
        ('final_layer_norm.weight', 'output/LayerNorm/gamma'),
        ('final_layer_norm.bias', 'output/LayerNorm/beta'),
        ('.', '/'),
    )

    def to_var_name(name):
        for patt, repl in iter(var_map):
            name = name.replace(patt, repl)
        return name
    import tensorflow as tf
    def create_tf_var(tensor:np.ndarray, name:str, session:tf.Session):
        tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
        tf_var = tf.get_variable(dtype=tf_dtype, shape=tensor.shape, name=name, initializer=tf.zeros_initializer())
        session.run(tf.variables_initializer([tf_var]))
        session.run(tf_var)
        return tf_var

    # Now let's copy all the weights.
    tf.reset_default_graph()
    with tf.Session() as session:
        state_dict = roberta.model.encoder.state_dict()
        for var_name in state_dict:
            tf_name = to_var_name(var_name)
            torch_tensor = state_dict[var_name].numpy()
            if any([x in var_name for x in tensors_to_transpose]):
                print("Transpose {}".format(tf_name))
                torch_tensor = torch_tensor.T
            tf_var = create_tf_var(tensor=torch_tensor, name=tf_name, session=session)
            tf.keras.backend.set_value(tf_var, torch_tensor)
            tf_weight = session.run(tf_var)
            print("Successfully created {}: {}".format(tf_name, np.allclose(tf_weight, torch_tensor)))

        # set token_type_embeddings
        # tf_name = 'bert/embeddings/token_type_embeddings'
        # torch_tensor = np.zeros((1, roberta.args.encoder_embed_dim))
        # tf_var = create_tf_var(tensor=torch_tensor, name=tf_name, session=session)
        # tf.keras.backend.set_value(tf_var, torch_tensor)
        # tf_weight = session.run(tf_var)
        # print("Successfully created {}: {}".format(tf_name, np.allclose(tf_weight, torch_tensor)))

        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(session, os.path.join(ckpt_dir, model_name.replace("-", "_") + ".ckpt"))

    # save config.json
    import json
    with open(os.path.join(ckpt_dir, 'bert_config.json'), 'w') as f:
        json.dump(config.__dict__, f)

    # save dict.txt
    with open(os.path.join(roberta_checkpoint_path, 'dict.txt'), 'r') as f:
        with open(os.path.join(ckpt_dir, 'dict.txt'), 'w') as tf:
            for line in f:
                tf.write(line)


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        default='roberta_large',
                        help="model name e.g. roberta_large")
    parser.add_argument("--cache_dir",
                        type=str,
                        default=r'./models/roberta.large',
                        help="Directory containing pytorch model")
    parser.add_argument("--tf_cache_dir",
                        type=str,
                        default=r'./models/roberta_large_fairseq',
                        help="Directory in which to save tensorflow model")
    args = parser.parse_args(raw_args)

    convert_roberta_checkpoint_to_tf(
        roberta_checkpoint_path=args.cache_dir,
        ckpt_dir=args.tf_cache_dir,
        model_name=args.model_name
    )


if __name__ == "__main__":
    main()
