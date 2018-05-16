'''
Makes an analytic dataset from the raw NYC 2016-2017 data;
Mostly just cleans the free text fields, including the ICD codes,
but it also pulls month and year out of the AdmitDay field.
'''
import numpy as np
import pandas as pd
import h5py

from tools.generic import gigs, text_to_int
from tools.text import read_ccs, clean_column, clean_ccs_column

# Importing and cleaning the data
records = pd.read_csv('~/data/syndromic/nyc_2016_2017.txt', sep='\t')

# Renaming the columns
records.columns = [
    'admit_ct', 'hospcode', 'id', 'date', 
    'time', 'age', 'gender', 'zip', 'admireason', 
    'cc', 'icd', 'dd', 'disdate', 'travel', 'moa', 
    'dispo', 'dischdt_ct', 'resp', 'ili', 'diar', 
    'vomit', 'asthma'
]

records = records.drop(['admit_ct', 'id', 'time', 'zip', 'admireason',
                        'disdate', 'travel', 'dd', 'dischdate'], axis=1)

# Casting the syndrome categories as single bytes (bits?)
records['ili'] = records['ili'].astype(np.uint8)
records['asthma'] = records['asthma'].astype(np.uint8)
records['vomit'] = records['vomit'].astype(np.uint8)
records['diar'] = records['diar'].astype(np.uint8)
records['resp'] = records['resp'].astype(np.uint8)

# Cleaning the text fields; easy fixes first
records['hospcode'] = [doc.lower() for doc in records['hospcode']]
records['age'] = [doc.replace('-', '') for doc in records['age']]
records['moa'] = [doc.replace(' ', '').lower() for doc in records['moa']]
records['gender'] = [doc.lower() for doc in records['gender'].astype(str)]
records['dispo'] = [doc.replace(' ', '').lower() 
                    for doc in records['dispo'].astype('string')]
icd = records['icd'].astype('string')
icd = [doc.replace('.', '') for doc in icd]
icd = [doc.split('|') for doc in icd]
icd = [[code.replace(' ', '') for code in doc] for doc in icd]
icd = [[code.lower() for code in doc] for doc in icd]
records['icd'] = pd.Series([' '.join(codes) for codes in icd])

# Doing the bigger cleaning on chief complaint and discharge diagnosis
records['cc'] = pd.Series(
    clean_column(
        records['cc'].astype('string'),
        numerals=False,
        remove_empty=False
        )
    )
records['dd'] = pd.Series(
    clean_column(
        records['dd'].astype('string'),
        numerals=False,
        remove_empty=False
        )
    )

# Making a month and date column from  admit date;
# turns out this is much faster than converting to datetimes with
# pd.to_datetime and then pulling out the month and year
records['month'] = pd.Series([int(date[0:2]) for date in records['date']])
records['year'] = pd.Series([int(date[6:10]) for date in records['date']])

# Importing the ICreD codes and exporting the conversions to CSV
icd10 = read_ccs('~/data/syndromic/ccs/ccs_icd10.csv')
icd9 = read_ccs('~/data/syndromic/ccs/ccs_icd9.csv')
icd10_dict = dict(list(zip(icd10['icd_code'], icd10['ccs_code'])))
icd9_dict = dict(list(zip(icd9['icd_code'], icd9['ccs_code'])))
icd_dict = icd9_dict.copy()
icd_dict.update(icd10_dict)

# Copying the ICD codes and converting them to CCS
ccs_strings = [[unicode(icd_dict.get(code)) 
                for code in visit] for visit in icd]
ccs_unique = [np.unique(codes) for codes in ccs_strings]
records['ccs'] = pd.Series([' '.join(codes) for codes in ccs_unique])

# Converting the text categorical variables to integers
moa = text_to_int(records['moa'])
hosp = text_to_int(records['hospcode'])
gender = text_to_int(records['gender'])

# Replacing the categorical values with the integer values
records['moa'] = pd.Series(moa['values'], dtype=np.uint16)
records['hospcode'] = pd.Series(hosp['values'], dtype=np.uint16)
records['gender'] = pd.Series(gender['values'], dtype=np.uint16)

# Writing a dictionary of the dictionaries to disk
vocab_out = h5py.File('data/code_vocab.hdf5', mode='w')
vocab_out['hosp'] = np.array([hosp['vocab'].keys(),
                              hosp['vocab'].values()],
                             dtype=str).transpose()
vocab_out['gender'] = np.array([gender['vocab'].keys(),
                              gender['vocab'].values()],
                             dtype=str).transpose()
vocab_out['moa'] = np.array([moa['vocab'].keys(),
                              moa['vocab'].values()],
                             dtype=str).transpose()
vocab_out.close()

# Isolating the records with good CC fields
good_cc = np.where([len(doc) != 0 for doc in records['cc']])[0]
good_cc_records = records.iloc[good_cc, :]

# Writing the whole files to CSV
records.to_csv('~/data/syndromic/nyc_clean.csv', index=False)
good_cc_records.to_csv('~/data/syndromic/good_cc_records.csv', index=False)
