# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:43:50 2019

@author: pg255026

build from bert-master
encode sentence using pooler layer
"""

import collections
import modeling
import tokenization
import tensorflow as tf
import args
import re

class InputExample(object):
    
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""
    
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids

def model_fn_builder(bert_config, init_checkpoint, layer_indexes):
    """Returns `model_fn` closure for Estimator."""
    
    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""
    
        unique_ids = features["unique_ids"]
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        input_type_ids = features["input_type_ids"]
    
        model = modeling.BertModel(
                config=bert_config,
                is_training=False,
                input_ids=input_ids,
                input_mask=input_mask,
                token_type_ids=input_type_ids)
    
        if mode != tf.estimator.ModeKeys.PREDICT:
            raise ValueError("Only PREDICT modes are supported: %s" % (mode))
    
        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
    
        # tf.logging.info("**** Trainable Variables ****")
        # for var in tvars:
        #     init_string = ""
        #     if var.name in initialized_variable_names:
        #         init_string = ", *INIT_FROM_CKPT*"
        #     tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)
        all_layers = model.get_all_encoder_layers()
        predictions = {
                "unique_id": unique_ids,
                }
    
        for (i, layer_index) in enumerate(layer_indexes):
            predictions["layer_output_%d" % i] = all_layers[layer_index]
        # default output pooler
        predictions["layer_output_pooler"] = model.get_pooled_output()
        
        from tensorflow.estimator import EstimatorSpec
        output_spec = EstimatorSpec(mode=mode, predictions=predictions)
        return output_spec
    return model_fn

def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
    
        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
    
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)
    
        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)
    
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
    
        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)
    
        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
    
        if ex_index < 5:
            tf.logging.info("*** Example ***")
            tf.logging.info("unique_id: %s" % (example.unique_id))
            tf.logging.info("tokens: %s" % " ".join(
                    [tokenization.printable_text(x) for x in tokens]))
            tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            tf.logging.info(
                    "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))
    
        features.append(
                InputFeatures(unique_id=example.unique_id,
                              tokens=tokens,
                              input_ids=input_ids,
                              input_mask=input_mask,
                              input_type_ids=input_type_ids))
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def input_fn_builder(features, seq_length):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""
    
    all_unique_ids = []
    all_input_ids = []
    all_input_mask = []
    all_input_type_ids = []
    
    for feature in features:
        all_unique_ids.append(feature.unique_id)
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_input_type_ids.append(feature.input_type_ids)
    
    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]
    
        num_examples = len(features)
    
        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
                "unique_ids":
                    tf.constant(all_unique_ids, shape=[num_examples], dtype=tf.int32),
                "input_ids":
                    tf.constant(
                        all_input_ids, shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "input_mask":
                    tf.constant(
                        all_input_mask,
                        shape=[num_examples, seq_length],
                        dtype=tf.int32),
                "input_type_ids":
                    tf.constant(
                        all_input_type_ids,
                        shape=[num_examples, seq_length],
                        dtype=tf.int32),
            })
    
        d = d.batch(batch_size=batch_size, drop_remainder=False)
        return d
    return input_fn
              
class BertEncoder:
    
    def __init__(self, args):
        from tensorflow.estimator import RunConfig, Estimator
        # load parameters
        self.layer_indexes = args.layer_indexes
        self.ckpt_name = args.ckpt_name
        self.config_name = args.config_name
        self.vocab_file = args.vocab_file
        self.do_lower_case = args.do_lower_case
        self.batch_size = args.batch_size
        self.max_seq_len = args.max_seq_len
        self.gpu_memory_fraction = args.gpu_memory_fraction
        self.xla = args.xla
        
        # load bert config & construct
        tf.logging.info("load bert config & construct ...")
        self.bert_config = modeling.BertConfig.from_json_file(self.config_name)
        model_fn = model_fn_builder(
                bert_config= self.bert_config,
                init_checkpoint=self.ckpt_name,
                layer_indexes=self.layer_indexes)
        
        # construct estimator
        tf.logging.info("load estimator ...")
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_memory_fraction
        config.log_device_placement = False
        if self.xla:
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        self.estimator = Estimator(model_fn=model_fn, config=RunConfig(session_config=config),
                         params={'batch_size': self.batch_size})
        
        self.tokenizer = tokenization.FullTokenizer(
                vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)
        tf.logging.info("initialization done.")
    
    def encode(self,input_sentences):
        return [sen['result']['layer_output_pooler'] for sen in self._predict(input_sentences)]
    
    def _predict(self,input_sentences):
        examples = self.read_examples(input_sentences)
        features = convert_examples_to_features(
                examples=examples, seq_length=self.max_seq_len, tokenizer=self.tokenizer)
        unique_id_to_feature = {}
        for feature in features:
            unique_id_to_feature[feature.unique_id] = feature
            
        input_fn = input_fn_builder(
                features=features, seq_length=self.max_seq_len)
        
        outputs_json = []
        for result in self.estimator.predict(input_fn, yield_single_examples=True):
            unique_id = int(result["unique_id"])
            feature = unique_id_to_feature[unique_id]
            output_json = collections.OrderedDict()
            output_json["linex_index"] = unique_id
            all_features = []
            for (i, token) in enumerate(feature.tokens):
                all_layers = []
                for (j, layer_index) in enumerate(self.layer_indexes):
                    layer_output = result["layer_output_%d" % j]
                    layers = collections.OrderedDict()
                    layers["index"] = layer_index
                    layers["values"] = [
                            round(float(x), 6) for x in layer_output[i:(i + 1)].flat
                            ]
                    all_layers.append(layers)
                features = collections.OrderedDict()
                features["token"] = token
                features["layers"] = all_layers
                all_features.append(features)
            output_json["features"] = all_features
            output_json["features_pooler"] = result["layer_output_pooler"]
            output_json["result"] = result
            outputs_json.append(output_json)
        return outputs_json
              

    
    def read_examples(self, input_sentences):
        """Read a list of `InputExample`s from a list of sentence instead of an input file."""
        examples = []
        unique_id = 0
        for line in  input_sentences:
            line = line.strip()
            text_a = None
            text_b = None
            m = re.match(r"^(.*) \|\|\| (.*)$", line)
            if m is None:
                text_a = line
            else:
                text_a = m.group(1)
                text_b = m.group(2)
            examples.append(
                    InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
            unique_id += 1
        return examples

if __name__ == '__main__':
    bert = BertEncoder(args)
    rst = bert.encode(['hello world']) # (['你好', '哈哈'])
    print(rst)
