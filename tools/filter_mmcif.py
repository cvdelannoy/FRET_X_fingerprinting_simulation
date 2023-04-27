import os
import sys
import traceback
import argparse
import pandas as pd
from Bio.PDB import MMCIF2Dict
import shutil

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(f'{__location__}/..')
from helpers import parse_input_dir, parse_output_dir

parser = argparse.ArgumentParser(description="""Filters *.cif AlphaFold files based on structuredness,
                                     N-terminal structuredness, and fit quality.""")
parser.add_argument('--in-dir', type=str, required=True)
parser.add_argument('--out-dir', type=str, required=True)
parser.add_argument('--copy_files', choices=[True, False], help="""If set to True, the filtered files
                                    will be copied to the output folder.""", type=bool, default=False)
parser.add_argument('--length_tresh', help='Maximum protein length.', type=int, default=600)
parser.add_argument('--fit-tresh',
                    help='Maximum AlphaFold fit quality defined in _ma_qa_metric_global.metric_value.',
                    type=float, default=0.8)
parser.add_argument('--N_struc_tresh', help='Structuredness of N-terminus in percent.', type=float, default=60)
parser.add_argument('--N_struc_len', help='Length of assesed N-terminus.', type=int, default=20)
parser.add_argument('--struc_types', choices='HBEGIPTSO',
                    help='DSSP code for the secondary structure.', type=str, default='HBEGIPTS')


dssp_dict = {"H": "HELX_RH_AL_P",
             "B": "STRN",
             "E": "STRN",
             "G": "HELX_RH_3T_P",
             "I": "HELX_RH_PI_P",
             "P": "HELX_LH_PP_P",
             "T": "TURN_TY1_P",
             "S": "BEND",
             "O": "OTHER", }



args = parser.parse_args()

out_dir = parse_output_dir(args.out_dir, clean=False)
error_log = out_dir + 'errors.log'
output_log = out_dir + 'output.log'
output_list = open(output_log, 'w')

args.struc_types = list(args.struc_types)


cif_list = parse_input_dir(args.in_dir, pattern='*.cif')

for cif_file in cif_list:
    try:
        mmcif_dict = MMCIF2Dict.MMCIF2Dict(cif_file)

        protein_len = len((mmcif_dict['_entity_poly.pdbx_seq_one_letter_code'][0]))
        fit_quality = float(mmcif_dict['_ma_qa_metric_global.metric_value'][0])

        cif_keys = ['_struct_conf.beg_auth_seq_id',
                    '_struct_conf.end_auth_seq_id',
                    '_struct_conf.conf_type_id']
        scnd_struc = pd.DataFrame(list(map(list, zip(*[mmcif_dict.get(key) for key in cif_keys]))),
                                  columns=['start', 'end', 'structure'])
        scnd_struc = scnd_struc.astype({'start': 'int', 'end': 'int'})
        scnd_struc['length'] = scnd_struc['end']-scnd_struc['start']+1

        struc_types = [dssp_dict.get(x) for x in args.struc_types]

        structurized_len = scnd_struc.query(f'end < {args.N_struc_len} \
                and structure in @struc_types')['length'].sum()

        N_struct_percent = structurized_len/args.N_struc_len * 100

        conditionals = [protein_len <= args.length_tresh,
                        fit_quality >= args.fit_tresh,
                        N_struct_percent >= args.N_struc_tresh
                        ]

        # save
        if all(conditionals):
            filename = os.path.split(cif_file)[-1]
            if args.copy_files == True:
                shutil.copy(cif_file, f'{out_dir}{filename}')
                output_list.writelines(filename)
            else:
                output_list.writelines(filename)
    except Exception as e:
        print(f'Something wrong with analysis of {cif_file}: line {e.__traceback__.tb_lineno}, {e}')
        traceback.print_exc()
        with open(error_log, 'a') as f:
            f.write(f'{cif_file}\t{e}\n')
