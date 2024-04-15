import os
import json
import glob
import shutil

datasets = {
    'train': [
        'spell-1.1',
        'time-1.9',
        'macchanger-1.6.0',
        'which-2.21',
        'ccd2cue-0.5',
        'wdiff-1.2.2',
        'nettle-3.9.1',
        'bool-0.2.2',
        'cppi-1.18',
        'hello-2.12.1',
        'libtool-2.4.7',
        'dap-3.10',
        'gss-1.0.4',
        'gzip-1.13',
        'units-2.23',
        'libtasn1-4.19.0',
        'libidn-2.3.7',
        'enscript-1.6.6',
        'gnudos-1.11.4',
        'libmicrohttpd-1.0.1',
        'cflow-1.7',
        'gsasl-2.2.1',
        'patch-2.7.6',
        'datamash-1.8',
        'libiconv-1.17',
        'gawk-5.3.0',
        'sed-4.9',
        'direvent-5.3',
        'cpio-2.15',
        'gdbm-1.23',
        'libunistring-1.2',
        'grep-3.11',
        'gnu-pw-mgr-2.7.4',
        'osip-5.3.1',
        'plotutils-2.6',
        'tar-1.35',
        'lightning-2.2.3',
        'gcal-4.1',
        'readline-8.2',
        'gmp-6.3.0',
        'coreutils-9.4',
        'recutils-1.9',
        'gsl-2.7.1',
        'binutils-2.42',
    ],
    'validation': [
        'gmp-6.3.0',
        'glpk-5.0',
        'xorriso-1.5.6',
    ],
    'evaluation': [
        'sharutils-4.15.2',
        'findutils-4.9.0',
        'inetutils-2.5'
    ]
}
with open('datasets/config.json', 'w') as fp:
    json.dump(datasets, fp, indent='\t')

for key, projects in datasets.items():
    print(f'{key}: {len(projects)}')
    for project in projects:
        dst = f'datasets/{key}'
        assert os.path.exists(dst)
        src = f'/data/lgy/GraduationProject/dataset/NEW/*/{project}'
        projectPath = glob.glob(src)
        assert len(projectPath) == 1
        src = projectPath[0]
        cmd = f'cp -r {src} {dst}'
        print(cmd)  # | parallel

