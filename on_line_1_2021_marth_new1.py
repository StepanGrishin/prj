#!/usr/bin/env python     #  для отладки и своего запуска редактируем скрипт в папке   /home/oleg/scripts/
# coding: utf-8           #    Этот скрипт выдает со страховкой  800 руб,  те не отрубает 800 руб
#                            Скрипт для ОН-лайн вторичников, используется все базы: БКИ+ФССП + внутренная кредитная история
#       наличие информации по клиенту не проверяется,тк это вторичник.  Первой моделью фильтруется поток, затем плохие и серая зона
#   дополнительно анализируются соответствуюшщими моделями файле: 
#   -  on_1_ver_2_basic.sav_9.sav – основная модель   
#   -  Model_bad.pkl          - Model_bad.pkl
#   -  Model_sr_4.pkl          - по "серой" зоне
#  Лоигика такая:  срабатывает первая модель, или одобряет или нет, если нет то опускаем сумму займа и пересчтываем веротяность
#  Если не помогает, то срабатывают следюущие модели и они опускают сумму займа    

#   При изменении моделей уделить внимание, что в качестве признака используется веротяность ОСНОВНОЙ модели, через  
#  дополнительный признак 'Prob_on_1_ver_2_basic.sav_9.pkl' , название конечно неказистое  (

#    Сейчас уменьшение суммы  займа выключено.  У вторичников  уменьшение суммы займа может приводить к снижениею вероятности, тк часто небольшие
#  суммы займа у берущих 2-3 займ, а больше сумму у берущих 4-6 займы  
#  ++++++++  Надо вывод переделать в следующий раз, сделать так, что "d={'score':score,'num_model':nu..."    выводился только ОДИН раз !!!

import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBClassifier
import json
import sys
import os

##################  Задание номера моделей  num_model   для записи в базу
num_model_1= 56           #  "Наш скоринг онлайн вторичники вер4"           Основная
name_file_1 ='on_1_ver_2_basic.sav_9.pkl'

#  сделаем 561  - для повышения суммы, а 562 - для понижения.

num_model_2= 57          #  "Наш скоринг онлайн вторичники вер4"           Серая зона  
name_file_2 ='Model_sr_4.pkl'

num_model_3= 58           #  "Наш скоринг онлайн вторичники вер4"             Плохиши
name_file_3 ='Model_bad.pkl'
##########################################
flag=0         #    flag=1  Печатать комменты,   flag=0  - не печатать 
flag_load=0    #    flag_load=1  - грузим модель с компа  иначе с сервера  
##########################################

initial_prob=''   

if flag==1:
    print('\n oleg/scripts Версия xgboost.__version__:',xgboost.__version__,'\n')

    print("Path is :",sys.argv [0])

    print('/home/oleg/scripts/on_line_1_2021_marth.py')

if flag_load==1 and 1==2:      #  берем даннные из файла  или  Строки вызова скрипта
        sys_argv='c:\\tmp\\PersonId.txt'
        
        table_main  = pd.read_csv(sys_argv ,sep='`', encoding='cp1251',low_memory=False) 
        if flag==1 and 1==2:
            print(table_main)

else:          #  берем даннные Из файла, а путь узнаем по API   (строка запроса)
    
    sys_argv=sys.argv[1]
        
    if flag==1:
        print('Берем данные из файла:',sys_argv)
        
    try:    
        with open(sys_argv) as f:
            todos = json.load(f)
            if flag==1:
                print('todos',todos)
        table_main = pd.DataFrame.from_dict(todos, orient='index') 
           
        table_main=table_main.T               #  раньше это было ниже , после IF    
            
    except:
        print('Не могу открыть файл:',sys_argv)    
# -----------------  вытаскиваю максимальную сумму доступную для заемщика в зависимости от числа займов  ----------------------------------------

MaxSum = table_main['ServiceData'].apply(lambda x: x.get("MaxSum")).values[0]
RequestAmount = table_main['ServiceData'].apply(lambda x: x.get("RequestAmount")).values[0]

if MaxSum == 0:
    MaxSum = 15000    

if flag==1:
    print('MaxSum:' , MaxSum)
# --------------------------------------------------------------------
#table_main.replace(to_replace='null', value=np.nan, inplace=True) 

#print('table_mainn1',table_main.head(1))  

if flag==1 and 1==2:                #  1  - печатать комменты
    Fin = open (sys_argv) 
    print(Fin.read())
    Fin.close()
    print('\n')            



if flag==1 and 1==2:                #  1  - печатать комменты    
    print('ТИпы данных:\n',table_main.shape, table_main.info())   
    print('table_main.head(1)\n',table_main.head(1))        #  тк дальше можем 2 раза выбирать разные столбцы     

#      Загруза основной модели и расчет вероятности
if flag_load!=1:     #  боевой вариант    берем файл с моделью с сервера
	path2=os.path.abspath(os.path.dirname(__file__))           #  где лежит исполняемый срипт и рядом сохраненная модель
	model_pkl = open(path2+'/'+name_file_1, 'rb')
else :       #  тестовый вариант   берем из файла на компе, когда модель запускается на компе
	path2='c:\\tmp\\'
	model_pkl = open(path2+name_file_1, 'rb')
	
model = pickle.load(model_pkl)        #      Загруза основной модели 

table_main['cred_appeal'] = table_main['prev_loans_cnt']/table_main['prev_drafts_cnt']

if 1==2:               # костыли для отладки
    del table_main['miAmount_90']
    table_main['miAmount_90']=np.nan
    del table_main['Cred_Deb_2']
    table_main['Cred_Deb_2']=np.nan

table=table_main[['avg_over_1_90',  'gibdd_180_count', 'Dola_RD_1', 'Duration', 'Count_12_len_num', 
                      'all_act_cred_3_1', 's_Amount_60','Count_4', 's_Amount_30', 'prolong', 'last_payment', 
                      'Cred_Deb_2', 'over_mfo', 'partner_type_3', 'max_hist_IPEndDate','max_Summa', 'miAmount_45', 
                      'sum_RIDDate_730', 'cred_date_last_2', 'sum_RIDDate_365', 'sum_debt_last', 'max_overdue',
            'sum_over_1080', 'min_hist_RIDDate', 'cred_365_count', 'day_overdue', 'min_hist_IPDate', 'Count_0_len_num',
        'gibdd_730_count', 'all_cred_rep_3', 'imush_180_count', 'age', 'cred_hist_1', 'sum_over_360', 'Count_C_len_num',
    's_Amount', 'delay60_1', 're_cred_1', 'all_act_cred_s', 'Count_456_len_num', 'max_hist_IPDate', 'max_hist_RIDDate',
    'cred_date_last_1', 'sum_debt_1', 'perc_amount', 'sum_mfo', 'delay30_1', 'a_sum_Credit_700', 'd_cnt_R', 'N_90_365',
    'cred_date_last', 'sum_over_1_89', 'sum_overdue', 'st46_4_365_count', 'gibdd_180_sum', 'F_iTA_30_365', 
    'st47_7_730_count','st46_3_365_count','cred_appeal', 'Amount']].astype(float).copy()

if RequestAmount != 0 and RequestAmount <= MaxSum:
    table['Amount'] = RequestAmount


#table['Amount']=5800              # для отладки
#print('5800       /home/oleg/scripts/on_line_1_2021_marth.py')

#for attrib in table.columns:
#    table[attrib]=table[attrib].astype(float)
      
if flag==1 and 1==1:        #   печать содержимого  TABLE  что идет в модель
    print('печать содержимого  TABLE  что идет в модель -------------------------------:',name_file_1)

    for index, row in table.iterrows():
        print (row)                   #
        
#    print(model)
    zz=model.predict_proba(table)          #  расчет вероятности
    print('вероятность\n',zz, table.shape)

score = round( 1- float(model.predict_proba(table)[:, 1]), 4)     #  расчет вероятности
                            #   тк Модель считает вероятность Дефолта, а нам нужна Вероятность Возврата   

initial_prob=score 
score_1=score  #   запоминает начальную веротяность, чтобы потом ее использовать при выборе модели  серые/плохие

table_main['Prob_on_1_ver_2_basic.sav_9.pkl']=score_1*100   #score_1         #  добавляем вероятность основной модели, тк может понадобиьтся 
                                                                 #  в моделях для плохишей и серой зоны
    
#table_main['prev_drafts_cnt']=0                #  костыль  ++++++++++++++++++++++++++++++++++++++++++++++++++++
   
if score >=0.85 and table['Amount'].values[0] +1000 <= MaxSum:           #    -------  очень хорошие, увелчиваем сумму и выводим в базу  

    score_85 = score           #  запоминаю начальные значения тк вдруг на первой итерации уйду ниже 85 %
    Amount_85 = table['Amount'].values[0]
    

    while table['Amount'].values[0] +1000 <= min(MaxSum, 2*RequestAmount) and score>=0.85:    #  именно больше 3000 , тк там дальше уменьшаем сумму и пересчитываем вероятность
        a = int(table['Amount'].values[0]/100)      #  вдруг число не круглое
        a = int(a*100 + 1000)

        table.at[0, 'Amount'] = a        
        
#        print('2 table[Amount].values[0]\n',table['Amount'].values[0])
        score = round( 1- float(model.predict_proba(table)[:, 1]), 4)     #  расчет вероятности
        if flag==1:
            print('score',score,'         table[Amount]:',table['Amount'].values[0]) 

    if a > RequestAmount: #(score >= 0.85)::
        score_85 = score
        Amount_85 = a
        num_model_1 = 561
    else:
        score_85 = initial_prob
        Amount_85 = RequestAmount
        num_model_1 = 56     

#сделаем 561  - для повышения суммы, а 562 - для понижения.
       
    d={'score':score_85,'num_model':num_model_1,'score2':'','num_model_2':'','Amount': Amount_85, 'initial_prob':initial_prob}   
    dj=json.dumps(d)
    print(dj)

elif (score >0.50 and score <0.85) or (score >=0.85 and RequestAmount + 1000 > MaxSum):           #    -------  хорошие, одрбряем  и выводим в базу
    if 1==2:
        print('{}, {}'.format(score,num_model1))
    else:
        d={'score':score,'num_model':num_model_1,'score2':'','num_model_2':'','Amount': '', 'initial_prob':initial_prob}   
        dj=json.dumps(d)
        print(dj)               #      
elif score <=0.50   and   1==1:          #    меняю сумму займа, чтоб изменилась вероятность,   "1==2" - выключение уменьшения суммы займа
 #   print('1 table[Amount]',table['Amount'].values[0], score )  
    a_new=''
    while table['Amount'].values[0]>= 3000 and score<=0.50:    #  именно больше 3000 , тк там дальше уменьшаем сумму и пересчитываем вероятность
        a=int(table['Amount'].values[0]/100)      #  вдруг число не круглое
        a=int(a*100-1000)
        table.at[0, 'Amount'] = a

#        print('2 table[Amount].values[0]\n',table['Amount'].values[0])
        score = round( 1- float(model.predict_proba(table)[:, 1]), 4)     #  расчет вероятности
        if flag==1:
            print('score',score,'         table[Amount]:',table['Amount'].values[0])

    if score >0.50:           #    -------  хорошие, одрбряем  и выводим в базу
        num_model_1=562   
        a_new=a

    d={'score':score,'num_model':num_model_1,'score2':'','num_model_2':'','Amount': a_new, 'initial_prob':initial_prob}   
    dj=json.dumps(d)
    print(dj) 

              
#  Здесь два варианта, или  score>50 после пересчета или сразу, или меньше - тогда вторая модель                  
#    ---------------------------------------------------------------------------------------------- серая зона   
#             те   серая  зона             и не нашли сумму с высокой веротяностью               
if score_1 >=0.30 and  score_1 <=0.50 and score<=0.50  and 1==2:   #   те это серая зона

    #      Загруза основной модели и расчет вероятности
    if flag_load!=1:     #  боевой вариант    берем файл с моделью с сервера
        path2=os.path.abspath(os.path.dirname(__file__))           #  где лежит исполняемый срипт и рядом сохраненная модель
        model_pkl = open(path2+'/'+name_file_2, 'rb')
    else :       #  тестовый вариант   берем из файла на компе, когда модель запускается на компе
        path2='c:\\tmp\\'
        model_pkl = open(path2+name_file_2, 'rb')

    model = pickle.load(model_pkl)       #      Загруза основной модели 
                

    table=table_main[['Count_0_len_num', 's_cnt_cred_90', 'partner_type_3', 
                  'last_payment', 's_Amount_90', 'miAmount_90', 'avg_over_1_90',
                 'd_cnt_12030', 'prolong_2', 'prolong', 'prolong_1', 'ti_hour', 'max_overdue_3', 
                 'max_overdue_2', 'max_overdue_1', 'max_overdue', 'prolong_3', 'Prob_on_1_ver_2_basic.sav_9.pkl',
                 'ti_day', 'ti_week', 'sum_overdue', 'ti_week_all', 'ti_month', 'ti_month_all', 'ti_quarter', 'ti_year',
                 'cred_date_last', 'cred_date_last_1', 'cred_date_last_2', 'day_overdue', 'sum_debt', 'sum_limit', 
                 'sum_debt_1', 'cred_type_3', 'partner_type_1', 'partner_type_2', 'cred_repay_1', 'cred_repay_2', 
                 'cred_repay_3', 'cred_activ_1', 'cred_activ_2', 'cred_activ_3', 'all_act_cred_3', 'all_act_cred_1',
                 'all_act_cred_3_1', 'all_act_cred_2', 'all_act_3_1_d', 'all_cred_rep_3', 'all_cred_rep_1', 
                 'all_cred_rep_13', 'all_cred_rep_11', 'all_rep_cred_1', 'all_act_cred', 'all_act_cred_s', 
                 'hist_days', 'sum_debt_last', 'Amount']].astype(float).copy()     
    

    if flag==1 and 1==2:        #   печать содержимого  TABLE  что идет в модель

        for row in table.iterrows():
            print (row)                   #(row['c1'], row['c2'])

        zz=model.predict_proba(table)          #  расчет вероятности
        print('вероятность:',zz)

    score = round( 1- float(model.predict_proba(table)[:, 1]), 4)     #  расчет вероятности

    if flag==1:
        print('\nmodel',name_file_2,'\tверотяность',score, '\ttable.shape:',table.shape)

    if score >0.50:           #    -------  хорошие, одрбряем  и выводим в базу
        if 1==2:
            print('{}, {}'.format(score,num_model1))
        else:
            d={'score':score_1,'num_model':num_model_1,'score2':score,'num_model_2':num_model_2,'Amount': ''}
            dj=json.dumps(d)
            print(dj)                                            #     1==2 - выключение снижение суммы займа   
    elif score <=0.50 and int(table['Amount'].values[0]/1000) >=3  and  1==2:    #  вероятность низка и желаемую сумму займа можно уменьшать 
        if flag==1:  
            print('1 table[Amount]',table['Amount'].values[0], score , model.predict_proba(table)   )  
    
        while table['Amount'].values[0]>= 3000 and score<=0.50:    #  именно больше 3000 , тк там дальше уменьшаем сумму и пересчитываем вероятность
            a=int(table['Amount'].values[0]/100)      #  вдруг число не круглое
            a=int(a*100-1000)
#            print(' 1 ',model.predict_proba(table)  , table['Amount'].values[0] )
            table.at[0, 'Amount'] = a
 #           print(' 2 ',model.predict_proba(table)  , table['Amount'].values[0]  )

            print('\n2 table[Amount].values[0]:',table['Amount'].values[0], a, '\ttable.shape:',table.shape,'\t',model.predict_proba(table) )
            score = round( 1- float(model.predict_proba(table)[:, 1]), 4)     #  расчет вероятности
    
            if flag==1:
                print('score',score,'         table[Amount]:',table['Amount'].values[0])
        
# вышли с цикла ,  или просто вышли  - веротяность низкам  или высокая и уменьшенная сумма

        d={'score':score_1,'num_model':num_model_1,'score2':score,'num_model_2':num_model_2,'Amount': a, 'initial_prob':initial_prob}   
        dj=json.dumps(d)
        print(dj) 

    else:         # вероятность низка и сумму займа уменьшать некуда, они и так меньше 3000 тыс
        d={'score':score_1,'num_model':num_model_1,'score2':score,'num_model_2':num_model_2,'Amount': '', 'initial_prob':initial_prob}   
        dj=json.dumps(d)
        print(dj) 


#    ---------------------------------------------------------------------------------------------- плохиши   
#  Здесь два варианта, или  score>50 после пересчета или сразу, или меньше - тогда вторая модель                  
#    те   плохиши       и не нашли сумму с высокой веротяностью               
if score_1 <0.30 and score<=0.50 and 1==2:    #   те это плохиши
    
    print('+++++++++++++++++++++++++++++++++')

    #      Загруза основной модели и расчет вероятности
    if flag_load!=1:     #  боевой вариант    берем файл с моделью с сервера
        path2=os.path.abspath(os.path.dirname(__file__))           #  где лежит исполняемый срипт и рядом сохраненная модель
        model_pkl = open(path2+'/'+name_file_3, 'rb')
    else :       #  тестовый вариант   берем из файла на компе, когда модель запускается на компе
        path2='c:\\tmp\\'
        model_pkl = open(path2+name_file_3, 'rb')

    model = pickle.load(model_pkl)       #      Загрузка модели 

    table=table_main[['Count_0_len_num', 's_Amount_60', 'miAmount_90',
                 's_Credit_befo_RD', 'Prob_on_1_ver_2_basic.sav_9.pkl', 'prolong_1', 
                 'prolong_2', 'max_overdue_2', 'prolong', 'max_overdue_3', 'max_overdue_1',
                 'max_overdue', 'day_overdue', 'prolong_3', 'ti_week', 'ti_hour', 'ti_day', 
                 'sum_limit', 'ti_week_all', 'ti_month', 'ti_month_all', 'ti_quarter', 'ti_year',
                 'cred_date_last', 'cred_date_last_1', 'cred_date_last_2', 'sum_debt_last', 'sum_overdue',
                 'test_1', 'sum_debt_1', 'sum_debt', 'cred_type_3', 'partner_type_1', 'partner_type_2',
                 'partner_type_3', 'cred_repay_1', 'cred_repay_2', 'cred_repay_3', 'cred_activ_1', 'cred_activ_2',
                 'cred_activ_3', 'all_act_cred_3', 'all_act_cred_1', 'all_act_cred_3_1', 'all_act_cred_2',
                 'all_act_3_1_d', 'all_cred_rep_3', 'all_cred_rep_1', 'all_cred_rep_13', 'all_cred_rep_11',
                 'all_rep_cred_1', 'all_act_cred', 'all_act_cred_s', 'hist_days_m', 'hist_days', 'dog_bad_1080',
                 'dog_bad', 'dog_bad_180', 'Count_9', 'Count_B', 'Count_C', 'Count_I', 'Count_R', 'Count_S',
                 'Count_Sp', 'Count_T', 'Count_U', 'Count_W', 'Count_01_len_num', 'Count_12_len_num',
                 'Count_456_len_num', 'Count_5B_len_num', 'Count_C_len_num', 'Count_Sp_len_num', 'N_record_IP',
                 'min_hist_IPDate', 'max_hist_IPDate', 'min_hist_RIDDate', 'max_hist_RIDDate',
                 'min_hist_IPEndDate', 'max_hist_IPEndDate', 'Count_8', 'Count_7', 'Count_6', 'sum_mfo',
                 'dog_bad_360', 'cred_type_1', 'sum_over_1080', 'sum_over_720', 'sum_over_360',
                 'sum_over_1_89', 'sum_over_90', 'sum_over_89_90', 'avg_over_1_90', 'sum_over_9000',
                 'over_mfo', 'Count_5']].astype(float).copy()      

    if 1==2:
        print('--------------------------------------------') 
        print(table.dtypes)  
        print('table.shape:',table.shape)

    if flag==1 and 1==2:        #   печать содержимого  TABLE  что идет в модель
        print('========= модель:',name_file_3,'\n')
        i=0
        for index, row in table.T.itertuples():
#        for index, row in table.T.iterrows():
            print (index,'\t', row)                  #
            
            
#        print(model)
        zz=model.predict_proba(table)          #  расчет вероятности
        print('вероятность\n',zz, table.shape)

    score = round( 1- float(model.predict_proba(table)[:, 1]), 4)     #  расчет вероятности

    if 1==2:
        print('{}, {}'.format(score,num_model_1))
    else:
        d={'score':score_1,'num_model':num_model_1,'score_2':score,'num_model_2':num_model_3,'Amount': '', 'initial_prob':initial_prob}   
        dj=json.dumps(d)
        print(dj)