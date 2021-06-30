import argparse, os, sys, re
from os.path import basename, dirname, realpath, abspath
import snakemake as sm
from itertools import chain
from jinja2 import Template

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(os.path.split(__location__)[0])

from helpers import parse_output_dir

def get_res_csv_names(in_dirs, resolutions, max_folds):
    res_csv_list = []
    latdir_dict = {}
    for dir_fn in in_dirs:
        bn = basename(realpath(dir_fn))
        tag_resn, labmod = re.search('(?<=tag)[A-Z]+', bn).group(0), re.search('(?<=labmod)[^_]+', bn).group(0)
        run_params = f'tag{tag_resn}_labmod{labmod}'
        latdir_dict[run_params] = dir_fn
        cur_csv_list = list(chain.from_iterable([[f'{res_dir}{run_params}_res{res}_{fold}.csv' for res in resolutions]
                        for fold in range(max_folds)]))
        res_csv_list.extend(cur_csv_list)
    return res_csv_list

parser = argparse.ArgumentParser(description='Run full random forest classification pipeline')
parser.add_argument('--in-dirs', required=True, nargs='+',
                    help='Input directories. Must be named as tagXX_labmodY (e.g. tagC_labmodPerfect,'
                         'or tagCK_labmodRegular)')
parser.add_argument('--cores', type=int, default=4)
parser.add_argument('--classifier', type=str, choices=['random_forest', 'boosted_tree', 'cross_correlation', 'cctree', 'cccombo', 'knn', 'svm'], default='random_forest',
                   help='Define type of classifier, may choose from random_forest, boosted_tree, cross_correlation, cctree, cccombo [default: random_forest]')
parser.add_argument('--original-dir', required=True,
                    help='dir containing original npz files used in constructing lattice models')
parser.add_argument('--max-folds', type=int, default=5)
parser.add_argument(' --define-good', type=float, default=0.5)
parser.add_argument('--out-dir', required=True, type=str,
                    help='output directory')
parser.add_argument('--res-range', type=int, nargs=3, default=[1, 21, 1],
                   help='start, stop, step for resolution range to plot')
args = parser.parse_args()

indir_basedirs = [abspath(dirname(dn)) if '/' in dn else dn for dn in args.in_dirs]
assert [indir_basedirs[0] == dn for dn in indir_basedirs]
indir_base = indir_basedirs[0]

# Make output folder structure
out_dir = parse_output_dir(args.out_dir)
res_dir = parse_output_dir(out_dir + 'classification_results')
cls_dir = parse_output_dir(out_dir + 'classifiers')
fp_dir = parse_output_dir(out_dir + 'parsed_fingerprints')

resolutions = list(range(*args.res_range))

res_csv_list = get_res_csv_names(args.in_dirs, resolutions, args.max_folds)
res_csv_list_best_resolution = get_res_csv_names(args.in_dirs, [resolutions[0]], args.max_folds)

if args.classifier == 'boosted_tree':
    train_cmd, classify_cmd = 'get_rf_classifier_xgb.py', 'classify_fingerprints_rf_xgb.py'
elif args.classifier == 'cross_correlation':
    train_cmd, classify_cmd = 'get_cc_classifier.py', 'classify_fingerprints_cc.py'
elif args.classifier == 'cctree':
    train_cmd, classify_cmd = 'get_cctree_classifier.py', 'classify_fingerprints_cctree.py'
elif args.classifier == 'cccombo':
    train_cmd, classify_cmd = 'get_cccombo_classifier.py', 'classify_fingerprints_cccombo.py'
elif args.classifier == 'knn':
    train_cmd, classify_cmd = 'get_knn_classifier.py', 'classify_fingerprints_knn.py'
elif args.classifier == 'svm':
    train_cmd, classify_cmd = 'get_svm_classifier.py', 'classify_fingerprints_svm.py'
else:
    train_cmd, classify_cmd = 'get_rf_classifier.py', 'classify_fingerprints_rf.py'

# Make snakemake workflow script
with open(f'{__location__}/fp_classification_template.sf', 'r') as fh: template_txt = fh.read()
sf_txt = Template(template_txt).render(
    __location__=__location__,
    res_csv_list=res_csv_list,
    res_csv_list_best_resolution=res_csv_list_best_resolution,
    indir_base=indir_base,
    original_dir=args.original_dir,
    out_dir=out_dir,
    fp_dir=fp_dir,
    res_dir=res_dir,
    cls_dir=cls_dir,
    resolutions=resolutions,
    define_good=args.define_good,
    train_cmd=train_cmd, classify_cmd=classify_cmd
)

sf_fn = f'{out_dir}analysis_pipeline.sf'
with open(sf_fn, 'w') as fh: fh.write(sf_txt)
sm.snakemake(sf_fn, cores=args.cores, dryrun=False, verbose=False)
