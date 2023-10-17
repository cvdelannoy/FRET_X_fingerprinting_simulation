import os
import sys
import traceback
import argparse
import pandas as pd
from Bio.PDB import MMCIF2Dict
import shutil
import requests
import csv

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(f'{__location__}/..')
from helpers import parse_input_dir, parse_output_dir

def str2bool(v) -> bool:
    """Converts string to bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description="""Filters *.cif AlphaFold files based on structuredness,
                                     N-terminal structuredness, and fit quality.""")
parser.add_argument('--in-dir', type=str, required=True)
parser.add_argument('--out-dir', type=str, required=True)
parser.add_argument('--copy_files', choices=[True, False], help="""If set to True, the filtered files
                                    will be copied to the output folder.""", type=str2bool, default=False)

parser.add_argument('--location', choices=[True, False],
                    help="""If set to True, together with "copy_files", the filtered files
                                    will be organized in the output folder, based on the subcellular
                                    location.""", type=str2bool, default=False)

parser.add_argument('--length_tresh', help='Maximum protein length.', type=int, default=600)
parser.add_argument('--fit-tresh',
                    help='Maximum AlphaFold pLDDT score defined in _ma_qa_metric_global.metric_value.',
                    type=float, default=0.7)
parser.add_argument('--N_struc_tresh', help='Structuredness of N-terminus in percent.', type=float, default=60)
parser.add_argument('--overall_struc_tresh', help='Structuredness of the protein in percent.', type=float, default=60)
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

header = ['accession', 'subcellularLocation']


def copy_files(file: str, dir: str) -> None:
    """Copies the given file to the given directory.
    The same files won't be copied to the directory,
    however, no exception will be printed."""

    filename = os.path.split(file)[-1]
    try:
        shutil.copy(file, f'{dir}{filename}')
    except shutil.SameFileError:
        pass


def get_subcellular_location(accession: str) -> list:
    """Get's the subcellular location based on the Uniprot acession code."""
    url = f'https://rest.uniprot.org/uniprotkb/search?query=accession:{accession}&fields=cc_subcellular_location'
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)

    try:
        if response.status_code == 200:

            data = response.json()
            try:
                locations_data = data['results'][0]['comments'][0]['subcellularLocations']
                subcellular_locations = [x['location']['value'] for x in locations_data]
                return subcellular_locations
            except:
                return ["None"]
        else:
            print(f"Error {response.status_code}: {response.reason}")
    except Exception as e:
        return e


if args.location == True:
    csv_file = open(f'{out_dir}subcellularLocation.csv', 'w', encoding='UTF8', newline='')
    writer = csv.writer(csv_file)
    writer.writerow(header)

for cif_file in cif_list:
    try:
        # %%
        # cif_file = r'C:\Users\makro\Desktop\fretx_tools\AF-Q32P42-F1-model_v4.cif'

        # %%
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
# %%
        struc_types = [dssp_dict.get(x) for x in args.struc_types]

        N_structurized_len = scnd_struc.query(f'end < {args.N_struc_len} \
                and structure in @struc_types')['length'].sum()
        N_struct_percent = N_structurized_len/args.N_struc_len * 100

        structurized_len = scnd_struc.query('structure in @struc_types')['length'].sum()
        overall_struc_percent = structurized_len/protein_len*100

        conditionals = [protein_len <= args.length_tresh,
                        fit_quality >= args.fit_tresh,
                        N_struct_percent >= args.N_struc_tresh,
                        overall_struc_percent >= args.overall_struc_tresh
                        ]

        # save
        if all(conditionals):
            filename = os.path.split(cif_file)[-1]
            # write csv with sucbellular location
            if args.location == True:
                accession = filename.split('-')[1]
                subcellular_locations = get_subcellular_location(accession)
                for location in subcellular_locations:
                    writer.writerow([accession, location])
            if args.copy_files == True:
                # write to the corresponding location folder
                if args.location == True:

                    for location in subcellular_locations:
                        location_out_dir = parse_output_dir(f'{args.out_dir}/location/{location}', clean=False)
                        copy_files(cif_file, dir=location_out_dir)
                        output_list.writelines(f'{filename}\n')

                else:
                    copy_files(cif_file, dir=out_dir)
                    output_list.writelines(f'{filename}\n')

            else:
                output_list.writelines(f'{filename}\n')
    except Exception as e:
        print(f'Something wrong with analysis of {cif_file}: line {e.__traceback__.tb_lineno}, {e}')
        traceback.print_exc()
        with open(error_log, 'a') as f:
            f.write(f'{cif_file}\t{e}\n')

# close the output files
output_list.close()
if args.location == True:
    csv_file.close()
