import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csvs = ['D:\Projetos\WattAI\Dataset/Building_59/Bldg59_clean data/ele.csv', #0
         'D:\Projetos\WattAI\Dataset/Building_59/Bldg59_clean data/site_weather.csv', #1
         'D:\Projetos\WattAI\Dataset/Building_59/Bldg59_clean data/zone_temp_sp_c.csv', #2
         'D:\Projetos\WattAI\Dataset/Building_59/Bldg59_clean data/zone_temp_sp_h.csv', #3
         'D:\Projetos\WattAI\Dataset/Building_59/Bldg59_clean data/zone_temp_interior.csv', #4
         'D:\Projetos\WattAI\Dataset/Building_59/Bldg59_clean data/zone_temp_exterior.csv', #5
         'D:\Projetos\WattAI\Dataset/Building_59/Bldg59_clean data/zone_co2.csv',   #6
         'D:\Projetos\WattAI\Dataset/Building_59/Bldg59_clean data/hp_hws_temp.csv', #7 # not used
         'D:\Projetos\WattAI\Dataset/Building_59/Bldg59_clean data/rtu_sa_t_sp.csv',    #8
         'D:\Projetos\WattAI\Dataset/Building_59/Bldg59_clean data/rtu_sa_t.csv',   #9  
         'D:\Projetos\WattAI\Dataset/Building_59/Bldg59_clean data/rtu_ra_t.csv',   #10
         'D:\Projetos\WattAI\Dataset/Building_59/Bldg59_clean data/rtu_oa_t.csv',   #11
         'D:\Projetos\WattAI\Dataset/Building_59/Bldg59_clean data/rtu_sa_fr.csv',  #12
         'D:\Projetos\WattAI\Dataset/Building_59/Bldg59_clean data/rtu_oa_damper.csv',  #13
         'D:\Projetos\WattAI\Dataset/Building_59/Bldg59_clean data/rtu_econ_sp.csv',  #14
         'D:\Projetos\WattAI\Dataset/Building_59/Bldg59_clean data/rtu_sa_p_sp.csv',  #15
         'D:\Projetos\WattAI\Dataset/Building_59/Bldg59_clean data/rtu_plenum_p.csv',  #16
         'D:\Projetos\WattAI\Dataset/Building_59/Bldg59_clean data/rtu_fan_spd.csv',    #17       
         'D:\Projetos\WattAI\Dataset/Building_59/Bldg59_clean data/uft_fan_spd.csv',    #18
         'D:\Projetos\WattAI\Dataset/Building_59/Bldg59_clean data/wifi.csv'] #19 # not used
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
df2 = pd.concat([df2['date'],df2['air_temp_set_1'],df2['air_temp_set_2']], axis=1)
df2_mean = pd.DataFrame(df2.mean(axis=1), columns= ['Outdoor_temp'] )
df2_mean = pd.concat([df2['date'], df2_mean],axis=1)

df_imp = pd.merge(df_imp,df2_mean , on='date', how='inner')

############################ df3

df3 = pd.read_csv(csvs[2])
df3['date'] = pd.to_datetime(df3['date'])
df3_mean = pd.DataFrame(df3['zone_061_cooling_sp'], columns= ['zone_temp_cooling'] )
# df3_mean = pd.DataFrame(df3.mean(axis=1), columns= ['zone_temp_cooling'] )
df3_mean['zone_temp_cooling'] = (df3_mean['zone_temp_cooling'] - 32) * 5/9 # convert temperature from F to C
df3_mean = pd.concat([df3['date'], df3_mean],axis=1)

df_imp = pd.merge(df_imp,df3_mean , on='date', how='inner')

######################### df4 

df4 = pd.read_csv(csvs[3])
df4['date'] = pd.to_datetime(df4['date'])
df4_mean = pd.DataFrame(df4['zone_061_heating_sp'], columns= ['zone_temp_heating'])
# df4_mean = pd.DataFrame(df4.mean(axis=1), columns= ['zone_temp_heating'])
df4_mean['zone_temp_heating'] = (df4_mean['zone_temp_heating'] - 32) * 5/9 # convert temperature from F to C
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

# df_10_mean = df_filtered_10.mean(axis=1)
# df_20_mean = df_filtered_20.mean(axis=1)
df_10_mean = df_filtered_10['cerc_templogger_4']
df_20_mean = df_filtered_20['cerc_templogger_4']

df_10_mean = df_10_mean.reset_index()
df_20_mean = df_20_mean.reset_index()
df_filtered_10 = df_filtered_10.reset_index()

mean_15 = (df_10_mean + df_20_mean) /2

mean_15 = mean_15.drop(['index'], axis=1)
mean_15 = pd.concat([df_filtered_10['date'] + pd.to_timedelta(5, unit='m'), mean_15], axis=1)
mean_15 = mean_15.iloc[:-1,:] # drop last row    (Nan)

# df_40_mean = df_filtered_40.mean(axis=1)
# df_50_mean = df_filtered_50.mean(axis=1)
df_40_mean = df_filtered_40['cerc_templogger_4']
df_50_mean = df_filtered_50['cerc_templogger_4']

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

# df6 = pd.read_csv(csvs[5])
# df6['date'] = pd.to_datetime(df6['date'])
# df6_mean = pd.DataFrame(df6.mean(axis=1), columns= ['indoor_temp_exterior'] )
# df6_mean['indoor_temp_exterior'] = (df6_mean['indoor_temp_exterior'] - 32) * 5/9 # convert temperature from F to C
# df6_mean = pd.concat([df6['date'], df6_mean],axis=1)

# df_imp = pd.merge(df_imp, df6_mean , on='date', how='inner')

########################## df7

df7 = pd.read_csv(csvs[6])
df7['date'] = pd.to_datetime(df7['date'])
df7_mean = pd.DataFrame(df7.mean(axis=1), columns= ['co2'] )
df7_mean = pd.concat([df7['date'], df7_mean],axis=1)

df_imp = pd.merge(df_imp,df7_mean , on='date', how='inner')

########################## df8

# df8 = pd.read_csv(csvs[18])
# # print(df8)
# df8['date'] = pd.to_datetime(df8['date'])
# df8_mean = pd.DataFrame(df8.mean(axis=1), columns= ['fan_speed'] )
# df8_mean = pd.concat([df8['date'], df8_mean],axis=1)

# df_imp = pd.merge(df_imp,df8_mean , on='date', how='inner')
# df_imp.fillna(method='ffill', inplace=True)

########################## df9 

df9 = pd.read_csv(csvs[17])
df9['date'] = pd.to_datetime(df9['date'])
df9_mean_1 = pd.DataFrame(df9.iloc[:,1:5].mean(axis=1), columns= ['supplyfan_speed'] )
df9_mean_2= pd.DataFrame(df9.iloc[:,5:9].mean(axis=1), columns= ['returnfan_speed'] )
df9_mean = pd.concat([df9['date'], df9_mean_1,df9_mean_2],axis=1)

df_imp = pd.merge(df_imp,df9_mean , on='date', how='inner')
df_imp.fillna(method='ffill', inplace=True)

# print(df_imp.isnull().sum())

# print(df_imp)

########################## df10

# df10 = pd.read_csv(csvs[7])
# df10['date'] = pd.to_datetime(df10['date'])
# df10_mean = pd.DataFrame(df10.mean(axis=1), columns= ['heat_water_supply_temp'])
# # convert temperature from F to C
# df10_mean['heat_water_supply_temp'] = (df10_mean['heat_water_supply_temp'] - 32) * 5/9
# df10_mean = pd.concat([df10['date'], df10_mean],axis=1)

# df_imp = pd.merge(df_imp,df10_mean , on='date', how='inner')

# # print date of last row
# print(df_imp.iloc[-1,0])

########################## df11

# df11 = pd.read_csv(csvs[8])
# df11['date'] = pd.to_datetime(df11['date'])
# df11_mean = pd.DataFrame(df11.mean(axis=1), columns= ['supply_air_temp_setpoint'])
# # convert temperature from F to C
# df11_mean['supply_air_temp_setpoint'] = (df11_mean['supply_air_temp_setpoint'] - 32) * 5/9
# df11_mean = pd.concat([df11['date'], df11_mean],axis=1)

# df_imp = pd.merge(df_imp,df11_mean , on='date', how='inner')


########################## df12

df12 = pd.read_csv(csvs[9])
df12['date'] = pd.to_datetime(df12['date'])
df12_mean = pd.DataFrame(df12.mean(axis=1), columns= ['supply_air_temp'])
# convert temperature from F to C
df12_mean['supply_air_temp'] = (df12_mean['supply_air_temp'] - 32) * 5/9
df12_mean = pd.concat([df12['date'], df12_mean],axis=1)
df12_mean = df12_mean.drop_duplicates(subset='date')
df_imp = pd.merge(df_imp,df12_mean , on='date', how='inner')


########################## df13

df13 = pd.read_csv(csvs[10])
df13['date'] = pd.to_datetime(df13['date'])
df13_mean = pd.DataFrame(df13.mean(axis=1), columns= ['return_air_temp'])
# convert temperature from F to C
df13_mean['return_air_temp'] = (df13_mean['return_air_temp'] - 32) * 5/9
df13_mean = pd.concat([df13['date'], df13_mean],axis=1)
df13_mean = df13_mean.drop_duplicates(subset='date')
df_imp = pd.merge(df_imp,df13_mean , on='date', how='inner')




########################## df14

# df14 = pd.read_csv(csvs[11])
# df14['date'] = pd.to_datetime(df14['date'])
# df14_mean = pd.DataFrame(df14.mean(axis=1), columns= ['outdoor_air_temp'])
# # convert temperature from F to C
# df14_mean['outdoor_air_temp'] = (df14_mean['outdoor_air_temp'] - 32) * 5/9
# df14_mean = pd.concat([df14['date'], df14_mean],axis=1)
# df14_mean = df14_mean.drop_duplicates(subset='date')
# df14_mean = df14_mean.drop_duplicates(subset='date')

# df_imp = pd.merge(df_imp,df14_mean , on='date', how='inner')


########################## df15

df15 = pd.read_csv(csvs[12])
df15['date'] = pd.to_datetime(df15['date'])
df15_mean = pd.DataFrame(df15.mean(axis=1), columns= ['filtered_air_flow_rate'])
# convert air flow rate from CFM to m3/h
df15_mean['filtered_air_flow_rate'] = df15_mean['filtered_air_flow_rate'] * 1.699
df15_mean = pd.concat([df15['date'], df15_mean],axis=1)
df15_mean = df15_mean.drop_duplicates(subset='date')

df_imp = pd.merge(df_imp,df15_mean , on='date', how='inner')


########################## df16

df16 = pd.read_csv(csvs[13])
df16['date'] = pd.to_datetime(df16['date'])
df16_mean = pd.DataFrame(df16.mean(axis=1), columns= ['outdoor_air_damper_position'])
df16_mean = pd.concat([df16['date'], df16_mean],axis=1)
df16_mean = df16_mean.drop_duplicates(subset='date')


df_imp = pd.merge(df_imp,df16_mean , on='date', how='inner')


########################## df17

# df17 = pd.read_csv(csvs[14])
# df17['date'] = pd.to_datetime(df17['date'])
# df17_mean = pd.DataFrame(df17.mean(axis=1), columns= ['economizer_setpoint'])
# df17_mean = pd.concat([df17['date'], df17_mean],axis=1)

# df_imp = pd.merge(df_imp,df17_mean , on='date', how='inner')


########################## df18

# df18 = pd.read_csv(csvs[15])
# df18['date'] = pd.to_datetime(df18['date'])
# df18_mean = pd.DataFrame(df18.mean(axis=1), columns= ['air_pressure_static_setpoint'])
# df18_mean = pd.concat([df18['date'], df18_mean],axis=1)

# df_imp = pd.merge(df_imp,df18_mean , on='date', how='inner')


########################## df19

# df19 = pd.read_csv(csvs[16])
# df19['date'] = pd.to_datetime(df19['date'])
# df19_mean = pd.DataFrame(df19.mean(axis=1), columns= ['plenum_air_pressure_at_floor'])
# df19_mean = pd.concat([df19['date'], df19_mean],axis=1)

# df_imp = pd.merge(df_imp,df19_mean , on='date', how='inner')

df_imp = df_imp.drop_duplicates(subset='date')


df_imp.to_csv("temp_data.csv", index=False)  

############################## Clean 

print(df_imp.min())