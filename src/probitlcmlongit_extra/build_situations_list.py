# This file is part of "probitlcmlongit_extra" which is released under GPL v3.
#
# Copyright (c) 2022-2025 Eric Alan Wayman <ericwaymanpublications@mailworks.org>.
#
# This program is FLO (free/libre/open) software: you can redistribute
# it and/or modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import pathlib
import json
import itertools

import numpy as np
from scipy.stats import multinomial

def generate_per_effect_M_j_s(K, rng):
    # the variable M_j_s should really be called per_effect_M_j_s
    M_j_s = list()
    M_j_pmf_values = [1/4, 1/2, 1/4]
    M_j_multinoulli_range = [4, 5, 6]
    # generate full main effect items
    u = 0; z = 0
    for k in range(1, K + 1):
        z = multinomial.rvs(n=1, p=M_j_pmf_values, size=1,
                            random_state=rng).argmax(axis=1).item()
        value_to_use = M_j_multinoulli_range[z]
        M_j_s.append(value_to_use)
    # generate full mixed effect items
    if K == 2: # add one per_effect item
        z = multinomial.rvs(n=1, p=M_j_pmf_values, size=1,
                            random_state=rng).argmax(axis=1).item()
        value_to_use = M_j_multinoulli_range[z]
        M_j_s.append(value_to_use)                
    elif K > 2:
        for i in range(1, K - 1): # 1 through K - 2
            for j in range(i + 1, K + 1): # i + 1 through K
                z = multinomial.rvs(n=1, p=M_j_pmf_values, size=1,
                                    random_state=rng).argmax(axis=1).item()
                value_to_use = M_j_multinoulli_range[z]
                M_j_s.append(value_to_use)
    return (M_j_s, rng)


def build_situation_info_files(sim_info_dir_name):
    # input "01_list_higherlevel_datagen_vals.json"
    # output "02_list_situations.json"
    # find q
    current_path = pathlib.Path.cwd()
    fpath = current_path.joinpath(sim_info_dir_name,
                                  "json_files",
                                  "01_fixed_vals.json")
    fixed_other_vals = json.loads(fpath.read_text())
    # load major dict for work
    current_path = pathlib.Path.cwd()
    fpath = current_path.joinpath(sim_info_dir_name,
                                  "json_files",
                                  "01_list_higherlevel_datagen_vals.json")
    m = json.loads(fpath.read_text())
    # do M_j_s related tasks
    dict_of_per_effect_M_j_s = dict()
    K_s = list()
    for L_k_s in m["L_k_s"]:
        mylen = len(L_k_s)
        if mylen not in K_s:
            K_s.append(mylen)
    seed_value = 0
    rng = np.random.default_rng(seed_value)
    for K in K_s:
        dict_of_per_effect_M_j_s[K], rng = generate_per_effect_M_j_s(int(K),
                                                                     rng)
    # prepare list of situation dicts
    l = list()
    # we sort the keys in reverse order to make rho appear first in cartesian
    # product
    my_keys = sorted(m.keys(), reverse=True)
    for x in my_keys:
        l.append(m[x])
    product_of_vals = itertools.product(*l)
    situations_list = list()
    i = 1
    for element in product_of_vals:
        situation = dict(zip(my_keys, element))
        situation["K"] = len(situation["L_k_s"])
        situation["per_effect_M_j_s"] = dict_of_per_effect_M_j_s[
            situation["K"]]
        situation["J"] = fixed_other_vals["q"] * \
            len(situation["per_effect_M_j_s"])
        situation["situation_num"] = i
        situations_list.append(situation)
        i = i + 1
    fname = "02_list_situations.json"
    fpath = current_path.joinpath(sim_info_dir_name, "json_files", fname)
    with fpath.open(mode='w') as f:
        json.dump(situations_list, f)    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_info_dir")
    args = parser.parse_args()
    build_situation_info_files(args.sim_info_dir)
