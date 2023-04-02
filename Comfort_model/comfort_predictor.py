"""

The goal of this script is to predict the comfort of a person in a room
based on the temperature and CO2 levels. Maybe other factors as well.

Run the script with the following command:
python ~/Desktop/Diogo/WattAI/WattAI/confort_predictor.py --input_file ~/Desktop/Diogo/WattAI/WattAI/dataset_building.csv 

Implemented by Diogo Araújo.

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pythermalcomfort as pc
from pythermalcomfort.models import pmv_ppd
from pythermalcomfort.utilities import met_typical_tasks, clo_individual_garments, v_relative, clo_dynamic

import argparse
from pathlib import Path

import os


def get_args_parser():
   
    parser = argparse.ArgumentParser('Confort predictor script', add_help=False)
    
    ## Add arguments here
    parser.add_argument('--output_dir', default='', help='path where to save, empty for no saving')
    parser.add_argument('--input_file', default='', help='path to input file')
    parser.add_argument('--building_type', default='office', help='type of building', 
                        choices=['office', 'residential', 'hotel', 'school', 'hospital', 
                                 'gym', 'retail', 'industrial', 'other'])
      
    return parser


def predict_pmv_ppd(df, args = None) -> dict[str, list]:
    """ Predicts the PMV and PPD values for a given dataset.
    
    PMV: Predicted Mean Vote | PPD: Predicted Percentage of Dissatisfied
    Check the pythermalcomfort documentation for more information.
    You can also check: https://www.simscale.com/blog/what-is-pmv-ppd/
    
    PMV -> [-3, 3] | PPD -> [0, 100]
    
    Args:
    df: Pandas dataframe containing the dataset.
    args: Additional arguments.

    Returns:
    pmv: List of PMV values.
    ppd: List of PPD values.
    """
    
    ### This needs to be ajusted to each specific case
    ## And I don't think this takes into account the number of persons in the room
    if args.building_type == 'office':
        activity = ['Typing',
                    'Filing, seated',
                    ]
        garments = ['Standard office chair',
                    'Double-breasted coat (thin)',
                    'Boots',
                    'Thick trousers',
                    'T-shirt',
                    ]
    elif args.building_type == 'hospital':
        activity = ['Walking about',
                    'Writing',
                    'Seated, heavy limb movement'
                    ]
        garments = ['Short-sleeve hospital gown',
                    'Boots',
                    'Calf length socks',
                    'Thick trousers',
                    ]
    elif args.building_type == 'school':
        activity = ['Typing',
                    'Filing, seated',
                    'Filing, standing',
                    'Writing',
                    'Walking about'
                    ]
        garments = ['Standard office chair',
                    'Overalls',
                    'Double-breasted coat (thin)',
                    'Boots',
                    'Calf length socks',
                    'Thick trousers',
                    'T-shirt',
                    ]
    elif args.building_type == 'residential':
        activity = ['Typing',
                    'Filing, seated',
                    'Filing, standing',
                    'Writing',
                    'Walking about'
                    ]
        garments = []
    elif args.building_type == 'gym':
        activity = []
        garments = []
    else:
        activity = []
        garments = []        
    
    
    ### Define the variables
    tdb = tr = df['indoor_temp_interior'].values
    tdb = np.array(tdb)
    tr = np.array(tr)    
    met = sum([met_typical_tasks[act] for act in activity])
    met = np.ones(len(df)) * met
    icl = sum([clo_individual_garments[garm] for garm in garments])
    icl = np.ones(len(df)) * icl
    v = np.ones(len(df)) * 0.3   # average air speed in m/s -> THIS NEEDS TO BE ADJUSTED TO THE RIGHT VALUE
    rh = np.ones(len(df)) * 50   # relative humidity in % ---> THIS NEEDS TO BE ADJUSTED TO THE RIGHT VALUE
    vr = v_relative(v=v, met=met)
    clo = clo_dynamic(clo=icl, met=met)
        

    ### Predict the PMV and PPD values
    results = pmv_ppd(tdb=tdb, tr=tr, vr=vr, rh=rh, met=met, clo=clo, standard='ASHRAE', units='SI') # See difference between ASHRAE and ISO !!
    
    return results['pmv'], results['ppd']

def main(args):

    ### Load the dataset
    file_name, file_extension = os.path.splitext(args.input_file)

    if file_extension == '.csv':
        df = pd.read_csv(args.input_file)
    elif file_extension == '.gz':
        df = pd.read_csv(args.input_file, compression='gzip')
        
    ### Predict the PMV and PPD values
    pmv, ppd = predict_pmv_ppd(df, args)
    print(f"Average PMV: {np.mean(pmv)} | Average PPD: {np.mean(ppd)}")

    ### Save the results
    df['pmv'] = pmv
    df['ppd'] = ppd
    df.to_csv("data_w_confort.csv", index=False)

    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Confort predictor script', parents=[get_args_parser()])
    args = parser.parse_args()
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    main(args)