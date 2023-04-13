import pandas as pd
from sklearn.preprocessing import StandardScaler


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

df_building = Predict_Comfort(args.validation_file)
df_synthetic = Predict_Comfort(args.train_file)
df_test = Predict_Comfort(args.test_file)
    
# Concatenate the two dataframes
df_train = pd.concat([df_synthetic, df_building], axis=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)