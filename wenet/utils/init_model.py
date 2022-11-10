# Copyright (c) 2022 Binbin Zhang (binbzha@qq.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from wenet.transducer.joint import TransducerJoint
from wenet.transducer.predictor import (ConvPredictor, EmbeddingPredictor,
                                        RNNPredictor)
from wenet.transducer.transducer import Transducer
from wenet.transformer.asr_model import ASRModel
from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.ctc import CTC
from wenet.transformer.decoder import BiTransformerDecoder, TransformerDecoder
from wenet.transformer.encoder import ConformerEncoder, TransformerEncoder
from wenet.utils.cmvn import load_cmvn


def init_model(configs):
    if configs['cmvn_file'] is not None:
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']
    vocab_size_py = configs['output_dim_py']

    encoder_type = configs.get('encoder', 'conformer')
    decoder_type = configs.get('decoder', 'bitransformer')

    if encoder_type == 'conformer':
        encoder_before = ConformerEncoder(input_dim,
                                          normalize_before=False,
                                          global_cmvn=global_cmvn,
                                          num_blocks=3,
                                   **configs['encoder_conf'])
        
        encoder_between = ConformerEncoder(256,
                                          normalize_before=False,
                                          global_cmvn=None,
                                          num_blocks=0,
                                   **configs['encoder_conf'])
        
        encoder = ConformerEncoder(256,
                                   normalize_before=True,
                                   global_cmvn=None,
                                   num_blocks=9,
                                   **configs['encoder_conf'])
        
        
    else:
        encoder = TransformerEncoder(input_dim,
                                     global_cmvn=global_cmvn,
                                     **configs['encoder_conf'])
    if decoder_type == 'transformer':
        decoder = TransformerDecoder(vocab_size, encoder.output_size(),
                                     **configs['decoder_conf'])
    else:
        assert 0.0 < configs['model_conf']['reverse_weight'] < 1.0
        assert configs['decoder_conf']['r_num_blocks'] > 0
        decoder = BiTransformerDecoder(vocab_size, encoder.output_size(),
                                       **configs['decoder_conf'])
    ctc = CTC(vocab_size, encoder.output_size())
    ctc_py = CTC(vocab_size_py,encoder.output_size())
    ctc1 = CTC(vocab_size, encoder.output_size())
    ## 要加CTC请在这里加


    dict_path = '/home/wenzhengchang/wenet_wzc/examples/aishell/s0/data/dict/lang_char.txt'
        # Load dict
    char_dict = {}
    with open(dict_path, 'r', encoding="utf-8") as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
       
        
    dict_pinyin_path = '/home/wenzhengchang/wenet_wzc/examples/aishell/s0/data/dict/lang_char_pinyin.txt'
    # Load dict_pinyin
    char_dict_pinyin = {}
    with open(dict_pinyin_path, 'r', encoding="utf-8") as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict_pinyin[int(arr[1])] = arr[0]


    # Init joint CTC/Attention or Transducer model
    if 'predictor' in configs:
        predictor_type = configs.get('predictor', 'rnn')
        if predictor_type == 'rnn':
            predictor = RNNPredictor(vocab_size, **configs['predictor_conf'])
        elif predictor_type == 'embedding':
            predictor = EmbeddingPredictor(vocab_size,
                                           **configs['predictor_conf'])
            configs['predictor_conf']['output_size'] = configs[
                'predictor_conf']['embed_size']
        elif predictor_type == 'conv':
            predictor = ConvPredictor(vocab_size, **configs['predictor_conf'])
            configs['predictor_conf']['output_size'] = configs[
                'predictor_conf']['embed_size']
        else:
            raise NotImplementedError(
                "only rnn, embedding and conv type support now")
        configs['joint_conf']['enc_output_size'] = configs['encoder_conf'][
            'output_size']
        configs['joint_conf']['pred_output_size'] = configs['predictor_conf'][
            'output_size']
        joint = TransducerJoint(vocab_size, **configs['joint_conf'])
        
        
        
        
        
        
        model = Transducer(vocab_size=vocab_size,
                           blank=0,
                           predictor=predictor,
                           encoder=encoder,
                           attention_decoder=decoder,
                           joint=joint,
                           ctc=ctc,
                           **configs['model_conf'])
    else:
        model = ASRModel(vocab_size=vocab_size,
                         vocab_size_py=vocab_size_py,
                         encoder_before=encoder_before,
                         encoder_between=encoder_between,
                         char_dict = char_dict,
                         char_dict_pinyin = char_dict_pinyin,
                         encoder=encoder,
                         decoder=decoder,
                         ctc=ctc,
                         ctc_py=ctc_py,
                         ctc1=ctc1,
                         **configs['model_conf'])
    return model
