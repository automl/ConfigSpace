#!/usr/bin/env python

##
# wrapping: A program making it easy to use hyperparameter
# optimization software.
# Copyright (C) 2013 Katharina Eggensperger and Matthias Feurer
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from argparse import ArgumentParser, FileType
from string import upper

from ConfigSpace.io import pb
from ConfigSpace.io import pyll
from ConfigSpace.io import pcs


__authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
__contact__ = "automl.org"


def main():
    # python convert.py --from SMAC --to TPE -f space.any -s space.else
    prog = "python convert.py"
    description = "Automatically convert a searchspace from one format to another"

    parser = ArgumentParser(description=description, prog=prog)

    parser.add_argument("--from", dest="conv_from", choices=['SMAC', 'Smac', 'smac',
                                                             'TPE', 'Tpe', 'tpe', 'hyperopt',
                                                             'SPEARMINT', 'Spearmint', 'spearmint'],
                        default="", help="Convert from which format?", required=True)
    parser.add_argument("--to", dest="conv_to", choices=['SMAC', 'Smac', 'smac',
                                                         'TPE', 'Tpe', 'tpe', 'hyperopt',
                                                         'SPEARMINT', 'Spearmint', 'spearmint'],
                        default="", help="Convert to which format?", required=True)
    parser.add_argument('input_file', nargs='?', type=FileType('r'))
    parser.add_argument("-s", "--save", dest="save", metavar="destination",
                        default="", help="Where to save the new searchspace?")

    args, unknown = parser.parse_known_args()

    # Unifying strings
    args.conv_to = upper(args.conv_to)
    args.conv_from = upper(args.conv_from)
    if args.conv_from == "HYPEROPT":
        args.conv_from = "TPE"
    if args.conv_to == "HYPEROPT":
        args.conv_to = "TPE"

    if args.input_file is None:
        raise ValueError("No input file given")

    read_options = {"SMAC": pcs.read,
                    "SPEARMINT": pb.read,
                    "TPE": pyll.read
                    }
    # First read searchspace
    print "Reading searchspace..."
    searchspace = read_options[args.conv_from](args.input_file)
    print "...done. Found %d params" % len(searchspace)

    write_options = {"SMAC": pcs.write,
                     "SPEARMINT": pb.write,
                     "TPE": pyll.write
                     }
    new_space = write_options[args.conv_to](searchspace)

    # No write it
    if args.save != "":
        output_fh = open(args.save, 'w')
        output_fh.write(new_space)
        output_fh.close()
    else:
        print new_space

if __name__ == "__main__":
    main()
