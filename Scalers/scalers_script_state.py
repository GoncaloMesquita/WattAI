import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle


######### main #########

##### Load data #####

df = pd.read_csv('dataset_building.csv')

print(df.columns)

df = df[['zone_temp_cooling', 
         'zone_temp_heating', 
         'supplyfan_speed',
         'returnfan_speed',
         'outdoor_air_damper_position',
         ]]

print(df.describe())


######### Standardization #########

# Apply feature scaling to the train and test datasets separately

df_building = pd.read_csv('dataset_building.csv')
df_synthetic = pd.read_csv('Synthetic_data/clean_synthetic_data.csv')

    
df_building = df_building[['Outdoor_temp', 
                    'indoor_temp_interior', 
                    'co2',
                    'supply_air_temp',
                    'return_air_temp',
                    'filtered_air_flow_rate'
                    ]]

df_synthetic = df_synthetic[['Outdoor_temp', 
                    'indoor_temp_interior', 
                    'co2',
                    'supply_air_temp',
                    'return_air_temp',
                    'filtered_air_flow_rate'
                    ]]

df_train = pd.concat([df_synthetic, df_building], axis=0)

train = df_train.values

sc = StandardScaler()
train = sc.fit_transform(train)

print(sc)

pickle.dump(sc, open('Scalers/state_scalers.pkl','wb'))
