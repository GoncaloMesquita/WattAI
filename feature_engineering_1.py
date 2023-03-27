import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csvs = ['doi_10.7941_D1N33Q__v6/Building_59/Bldg59_clean data/ele.csv',
         'doi_10.7941_D1N33Q__v6/Building_59/Bldg59_clean data/site_weather.csv',
         'doi_10.7941_D1N33Q__v6/Building_59/Bldg59_clean data/zone_temp_sp_c.csv',
         'doi_10.7941_D1N33Q__v6/Building_59/Bldg59_clean data/zone_temp_sp_h.csv',
         'doi_10.7941_D1N33Q__v6/Building_59/Bldg59_clean data/zone_temp_interior.csv',
         'doi_10.7941_D1N33Q__v6/Building_59/Bldg59_clean data/zone_temp_exterior.csv',
         'doi_10.7941_D1N33Q__v6/Building_59/Bldg59_clean data/zone_co2.csv', 
         'doi_10.7941_D1N33Q__v6/Building_59/Bldg59_clean data/rtu_fan_spd.csv', 
         'doi_10.7941_D1N33Q__v6/Building_59/Bldg59_clean data/uft_fan_spd.csv', 
         'doi_10.7941_D1N33Q__v6/Building_59/Bldg59_clean data/wifi.csv']
# data = pd.DataFrame()
# for proj in csvs:
#     df1 = pd.read_csv(proj, index_col=0, header=0)
#     # data.append(df1 )
#     data = pd.concat([data, df1], axis=1)
#     # print(data)
#     print(proj)

# print(data)

########################## df1

df1 = pd.read_csv(csvs[0])
df1['date'] = pd.to_datetime(df1['date']) 
df1_sum = pd.DataFrame(df1['hvac_N'] + df1['hvac_S'], columns= ['energy_hvac'] )
df_imp = pd.concat([df1['date'],df1_sum], axis=1)

########################## df2

df2 = pd.read_csv(csvs[1])
df2['date'] = pd.to_datetime(df2['date']) 
df2 = pd.concat([df2['date'],df2['air_temp_set_1'],df2['air_temp_set_1']], axis=1)
df2_mean = pd.DataFrame(df2.mean(axis=1), columns= ['Outdoor_temp'] )
df2_mean = pd.concat([df2['date'], df2_mean],axis=1)

df_imp = pd.merge(df_imp,df2_mean , on='date', how='inner')

############################ df3

df3 = pd.read_csv(csvs[2])
df3['date'] = pd.to_datetime(df3['date'])
df3_mean = pd.DataFrame(df3.mean(axis=1), columns= ['zone_temp_cooling'] )
df3_mean = pd.concat([df3['date'], df3_mean],axis=1)

df_imp = pd.merge(df_imp,df3_mean , on='date', how='inner')

######################### df4 

df4 = pd.read_csv(csvs[3])
df4['date'] = pd.to_datetime(df4['date'])
df4_mean = pd.DataFrame(df4.mean(axis=1), columns= ['zone_temp_heating'])
df4_mean = pd.concat([df4['date'], df4_mean],axis=1)

df_imp = pd.merge(df_imp,df4_mean , on='date', how='inner')

############################ df5

df5 = pd.read_csv(csvs[4])
df5['date'] = pd.to_datetime(df5['date'])
df5 = df5.drop(df5.index[13])
df_filtered_10 = df5[(df5['date'].dt.minute == 10)] #Get the all the data points that have 10 min
df_filtered_20 = df5[df5['date'].dt.minute == 20] #Get the all the data points that have 20 min

df_filtered_40 = df5[df5['date'].dt.minute == 40] #Get the all the data points that have 40 min
df_filtered_50 = df5[df5['date'].dt.minute == 50] #Get the all the data points that have 50 min

df_10_mean = df_filtered_10.mean(axis=1)
df_20_mean = df_filtered_20.mean(axis=1)

df_10_mean = df_10_mean.reset_index()
df_20_mean = df_20_mean.reset_index()
df_filtered_10 = df_filtered_10.reset_index()

mean_15 = (df_10_mean + df_20_mean) /2

mean_15 = mean_15.drop(['index'], axis=1)
mean_15 = pd.concat([df_filtered_10['date'] + pd.to_timedelta(5, unit='m'), mean_15], axis=1)
mean_15 = mean_15.iloc[:-1,:] # drop last row    (Nan)

df_40_mean = df_filtered_40.mean(axis=1)
df_50_mean = df_filtered_50.mean(axis=1)

df_40_mean = df_40_mean.reset_index()
df_50_mean = df_50_mean.reset_index()
df_filtered_40 = df_filtered_40.reset_index()

mean_45 = (df_40_mean + df_50_mean) /2

mean_45 = mean_45.drop(['index'], axis=1)
mean_45 = pd.concat([df_filtered_40['date'] + pd.to_timedelta(5, unit='m'), mean_45], axis=1)
mean_45 = mean_45.iloc[:-1,:] # drop last row (Nan)

df5_mean = pd.DataFrame(df5.mean(axis=1), columns= ['indoor_temp_interior'] )
df5_mean = pd.concat([df5['date'], df5_mean],axis=1)

df5_mean = pd.merge(df5_mean ,mean_45, how='outer', on='date', sort=True )

replacement_col = df5_mean.iloc[:,2]
df5_mean['indoor_temp_interior'].fillna(replacement_col, inplace=True)
df5_mean = df5_mean.drop(0, axis=1)

df5_mean = pd.merge(df5_mean ,mean_15, how='outer', on='date', sort=True )

replacement_col1 = df5_mean.iloc[:,2]
df5_mean['indoor_temp_interior'].fillna(replacement_col1, inplace=True)
df5_mean = df5_mean.drop(0, axis=1)
df_imp = pd.merge(df_imp,df5_mean , on='date', how='inner')

####################### df6  #################

df6 = pd.read_csv(csvs[5])
df6['date'] = pd.to_datetime(df6['date'])
df6_mean = pd.DataFrame(df6.mean(axis=1), columns= ['indoor_temp_exterior'] )
df6_mean = pd.concat([df6['date'], df6_mean],axis=1)

df_imp = pd.merge(df_imp, df6_mean , on='date', how='inner')

########################## df7

df7 = pd.read_csv(csvs[6])
df7['date'] = pd.to_datetime(df7['date'])
df7_mean = pd.DataFrame(df7.mean(axis=1), columns= ['co2'] )
df7_mean = pd.concat([df7['date'], df7_mean],axis=1)

df_imp = pd.merge(df_imp,df7_mean , on='date', how='inner')

########################## df8

df8 = pd.read_csv(csvs[8])
# print(df8)
df8['date'] = pd.to_datetime(df8['date'])
df8_mean = pd.DataFrame(df8.mean(axis=1), columns= ['fan_speed'] )
df8_mean = pd.concat([df8['date'], df8_mean],axis=1)

df_imp = pd.merge(df_imp,df8_mean , on='date', how='inner')
df_imp.fillna(method='ffill', inplace=True)

##########################3 df9 

df9 = pd.read_csv(csvs[7])
df9['date'] = pd.to_datetime(df9['date'])
df9_mean_1 = pd.DataFrame(df9.iloc[:,1:5].mean(axis=1), columns= ['supplyfan_speed'] )
df9_mean_2= pd.DataFrame(df9.iloc[:,5:9].mean(axis=1), columns= ['returnfan_speed'] )
df9_mean = pd.concat([df9['date'], df9_mean_1,df9_mean_2],axis=1)

df_imp = pd.merge(df_imp,df9_mean , on='date', how='inner')
df_imp.fillna(method='ffill', inplace=True)

# print(df_imp.isnull().sum())

print(df_imp)



df_imp.to_csv("dataset_building.csv", index=False)  