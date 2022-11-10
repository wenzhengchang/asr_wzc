# Copyright (c) 2020 Johns Hopkins University (Shinji Watanabe)
#               2020 Northwestern Polytechnical University (Pengcheng Guo)
#               2020 Mobvoi Inc (Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Swish() activation function for Conformer."""

import torch
from pypinyin import pinyin


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
        char_dict_pinyin[arr[0]] = int(arr[1])
        
        
        
        
test = 2


a = char_dict[test]

b = pinyin(a)

c = char_dict_pinyin[b[0][0]]

print(c)
        
    
