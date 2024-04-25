# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from unicore.data import Dictionary

def parse_select_atom(dictionary, select_atom):
    
    if select_atom == 'All':
        return ['All']
    else:
        return dictionary.vec_index(select_atom.split("&"))

