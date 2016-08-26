# # Copyright (c) 2014-2016, ConfigSpace developers
# # Matthias Feurer
# # Katharina Eggensperger
# # and others (see commit history).
# # All rights reserved.
# #
# # Redistribution and use in source and binary forms, with or without
# # modification, are permitted provided that the following conditions are met:
# #     * Redistributions of source code must retain the above copyright
# #       notice, this list of conditions and the following disclaimer.
# #     * Redistributions in binary form must reproduce the above copyright
# #       notice, this list of conditions and the following disclaimer in the
# #       documentation and/or other materials provided with the distribution.
# #     * Neither the name of the <organization> nor the
# #       names of its contributors may be used to endorse or promote products
# #       derived from this software without specific prior written permission.
# #
# # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# # ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# # WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# # DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# # DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# # (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# # LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# # ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# # SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# from argparse import ArgumentParser, FileType
# from string import upper
#
# from ConfigSpace.io import pb
# from ConfigSpace.io import pyll
# from ConfigSpace.io import pcs
#
#
# __authors__ = ["Katharina Eggensperger", "Matthias Feurer"]
# __contact__ = "automl.org"
#
#
# def main():
#     # python convert.py --from SMAC --to TPE -f space.any -s space.else
#     prog = "python convert.py"
#     description = "Automatically convert a searchspace from one format to another"
#
#     parser = ArgumentParser(description=description, prog=prog)
#
#     parser.add_argument("--from", dest="conv_from", choices=['SMAC', 'Smac', 'smac',
#                                                              'TPE', 'Tpe', 'tpe', 'hyperopt',
#                                                              'SPEARMINT', 'Spearmint', 'spearmint'],
#                         default="", help="Convert from which format?", required=True)
#     parser.add_argument("--to", dest="conv_to", choices=['SMAC', 'Smac', 'smac',
#                                                          'TPE', 'Tpe', 'tpe', 'hyperopt',
#                                                          'SPEARMINT', 'Spearmint', 'spearmint'],
#                         default="", help="Convert to which format?", required=True)
#     parser.add_argument('input_file', nargs='?', type=FileType('r'))
#     parser.add_argument("-s", "--save", dest="save", metavar="destination",
#                         default="", help="Where to save the new searchspace?")
#
#     args, unknown = parser.parse_known_args()
#
#     # Unifying strings
#     args.conv_to = upper(args.conv_to)
#     args.conv_from = upper(args.conv_from)
#     if args.conv_from == "HYPEROPT":
#         args.conv_from = "TPE"
#     if args.conv_to == "HYPEROPT":
#         args.conv_to = "TPE"
#
#     if args.input_file is None:
#         raise ValueError("No input file given")
#
#     read_options = {"SMAC": pcs.read,
#                     "SPEARMINT": pb.read,
#                     "TPE": pyll.read
#                     }
#     # First read searchspace
#     print "Reading searchspace..."
#     searchspace = read_options[args.conv_from](args.input_file)
#     print "...done. Found %d params" % len(searchspace)
#
#     write_options = {"SMAC": pcs.write,
#                      "SPEARMINT": pb.write,
#                      "TPE": pyll.write
#                      }
#     new_space = write_options[args.conv_to](searchspace)
#
#     # No write it
#     if args.save != "":
#         output_fh = open(args.save, 'w')
#         output_fh.write(new_space)
#         output_fh.close()
#     else:
#         print new_space
#
# if __name__ == "__main__":
#     main()
