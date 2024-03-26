# Copyright (c) 2014-2016, ConfigSpace developers
# Matthias Feurer
# Katharina Eggensperger
# and others (see commit history).
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the <organization> nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from __future__ import annotations

from pathlib import Path

import pytest

import ConfigSpace
import ConfigSpace.read_and_write.pcs as pcs_parser
import ConfigSpace.read_and_write.pcs_new as pcs_new_parser
import ConfigSpace.util

this_file = Path(__file__).absolute().resolve()
this_directory = this_file.parent
configuration_space_path = (
    (this_directory.parent / "test_searchspaces").absolute().resolve()
)
pcs_files = list(Path(configuration_space_path).glob("*.pcs"))


@pytest.mark.parametrize("pcs_file", pcs_files)
def test_autosklearn_space(pcs_file: Path):
    try:
        with pcs_file.open("r") as fh:
            cs = pcs_parser.read(fh)
    except Exception:
        with pcs_file.open("r") as fh:
            cs = pcs_new_parser.read(fh)

    default = cs.get_default_configuration()
    cs._check_configuration_rigorous(default)
    for i in range(10):
        neighborhood = ConfigSpace.util.get_one_exchange_neighbourhood(default, seed=i)

        for shuffle, n in enumerate(neighborhood):
            n.is_valid_configuration()
            cs._check_configuration_rigorous(n)
            if shuffle == 10:
                break

    print(cs)
    # Sample a little bit
    for i in range(10):
        cs.seed(i)
        for c in cs.sample_configuration(size=5):
            c.is_valid_configuration()
            cs._check_configuration_rigorous(c)
            neighborhood = ConfigSpace.util.get_one_exchange_neighbourhood(c, seed=i)

            for shuffle, n in enumerate(neighborhood):
                n.is_valid_configuration()
                cs._check_configuration_rigorous(n)
                if shuffle == 20:
                    break
