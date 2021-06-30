import sys
import urllib.parse
import urllib.request
from time import sleep
import requests

pdb_list_fn = sys.argv[1]

with open(pdb_list_fn, 'r') as fh: pdb_list = fh.readlines()


url = 'https://www.uniprot.org/uploadlists/'

params = {
'from': 'PDB_ID',
'to': 'ACC',
'format': 'tab',
'query': ' '.join(pdb_list)
}
sleep(2)
req_obj = requests.get(url, params=params)
req_list = req_obj.content.decode('utf-8').split('\n')[1:]
req_list = [r for r in req_list if '\t' in r]
up_list = [r.split('\t')[1] for r in req_list]
print('\n'.join(up_list))
