#!/usr/bin/env python
# coding: utf-8

#                            Скрипт для ОН-лайн с 0 погашенным, используются базы: Анкета + БКИ + Скорбалл ЭФ
#                            Заявка скорится 3 моделями, потом берется из среднее арифметическое и на его основании принимается конечное решение  
#                                                                                                                  
#   

import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import json
import sys
import os
import math

##################  Задание номера моделей  num_model   для записи в базу
num_model1=67            #  модель со скор-баллом Эквифакса
                         #   48  пишется, когда ничего не сработало и тогда веротяность = 0.01
num_model2=62            #   Логистическая регрессия

num_model3=63            #   "XGBoost   модель Наташи от ноября 2021"
                         #   ниже по num_model3  определяю  Уровень  одобрения ДЛЯ изменения суммы
num_model_res=65         # результирующая модель блендинга 
#сделаем 631  - для повышения суммы, а 632 - для понижения.

#   модель нет данных - 48  осталась,тк она основана на июньском XGBoost

 
model1=1        # 0 - Модель выключена,  1 - модель включена    XGBoost   июнь
model2=1	# 0 - Модель выключена,  1 - модель включена
model3=1	# 0 - Модель выключена,  1 - модель включена
model4=1	# 0 - Модель выключена,  1 - модель включена    XGBoost   ноябрь
model_res=1 # 0 - Модель выключена,  1 - модель включена
     #   и внизу поменять !!!!!!!!!
flag1=1        #  те сумма флагов ==2     если 3 то включена модель по трем источникам, а если например 4 - то выключена
##########################################
flag=0         #    flag=1  Печатать комменты,   flag=0  - не печатать 
##########################################
if flag==1:
    print('\n oleg/scripts Версия xgboost.__version__:',xgboost.__version__,'\n')

    #print("Path is :",sys.argv [0])


if 1==2:       #  берем даннные из файла  или  Строки вызова скрипта
        sys_argv='c:\\tmp\\PersonId.txt'
        
        with open(sys_argv) as f:
            todos = json.load(f)

else:          #  берем даннные Из файла, а путь узнаем по API   (строка запроса)
    
    sys_argv=sys.argv[1]
 #   print('sys_argv:',sys_argv)
        
    if flag==1:
        print('Берем данные из файла:',sys_argv)
    try:    
        with open(sys_argv) as f:
            todos = json.load(f)
            if flag==1:
                print('что в файле todos',todos)
    except:
        print('Не могу открыть файл:',sys_argv)
#    for section, commands in todos.items():
#        print(section, commands)

if flag==1 and 1==2:   
    f = open(sys_argv)
    f.read()
    print('f:')

table = pd.DataFrame.from_dict(todos, orient='index')

if flag==1 and 1==2:
    print('---------------------------------------------------------------------')
    print(table.to_string())


if flag==1 and 1==2:                #  1  - печатать комменты
    Fin = open (sys_argv) 
    print(Fin.read())
    Fin.close()
    print('\n')


table=table.T  

#   Borders - словарь где все Номера моделей и все значения  границ  - уровней Одобрения
Borders=table['Borders'].values[0]        

border = Borders.get(str(num_model3))/100    #   Вытаскиваю  Порог конкретной модели ДЛЯ изменения суммы

#   PolicyFee - сумма страховки, которая включена в во входжную сумму

PolicyFee= table['ServiceData'].values[0].get(str('PolicyFee'))

if flag==1:
    print('border:' , border, ' of', num_model3, '  PolicyFee:',PolicyFee)

table['countIssuedFor90Days1_sum'] = table['countIssuedFor90Days_sum']    

df=table.copy()        #  для логистической регрессии

table1=table.copy()       #  для  новой модели

if flag==1 and 1==1:                #  1  - печатать комменты    
    print('ТИпы данных:\n',table.shape, table.info())     

#flagFC=todos['flagFC']
#flagSP=todos['flagSP']
flagEF=todos['flagEF']
Amount_3=''                    #    измененная сумма , или вверх , или вниз

if flag==1:
    print( 'flagEF:', flagEF )



#   сделаю так, если нет ответа от Эквифакса, то просто присваиваю веротность = 0.01   и модель №  48
#  ---------------------------------------------------------------------------------   XGBoost  июнь  -----------------
                           #   и здесь поменять !!!!!!!!!   = flag1
if 1==1:    
    if 1==1:     #  боевой вариант    берем файл с моделью с сервера
        path2=os.path.abspath(os.path.dirname(__file__))           #  где лежит исполняемый срипт и рядом сохраненная модель
        model_pkl = open(path2+'/'+'on_0_4134_Ew.pkl', 'rb')            #  проверяем ТОТ-ли файл грузится  с сохраненной моделью ??!!!!
    else :       #  тестовый вариант   берем из файла на компе, когда модель запускается на компе
        path2='c:\\tmp\\'
        model_pkl = open(path2+'on_0_4134_Ew.pkl', 'rb')                #  проверяем ТОТ-ли файл грузится  с сохраненной моделью ??!!!! 
        
    model = pickle.load(model_pkl) 

    table1['interest_month_mod']=table1['interest_month_mod'].fillna(-999999)
#  Education_v 	для Наташиной модели в питоновском скрипте надо сделать преобразование в  Education_v,   высшее = 1 , иначе = 2										
    table1['Education_v']=table1['Education'].apply(lambda x: 1 if x=='высшее' else 2)

    table1['OrganizationPhone_optb100']=table1['OrganizationPhone'].apply(lambda x: 1 if str(x)[:5]=='+7 (9' else 3 if str(x)[:5]=='+7 (4' else 2)


    table1['Post'] = table1['Post'].fillna('Другое')
  
    
    table1['Post_m']=table1['Post'].map({'Рабочий': 0.442, 'Водитель': 0.442, 'Служащий/Специалист': 0.366, 'Начальник/руководитель': 0.357, 'Руководитель компании': 0.357, 'Другое': 0.41, 'Военные': 0.42}) 
    table1['MonthlyIncome'] = table1['MonthlyIncome'].astype('float').apply(lambda x: 150000 if x > 150000 else x)
    
    table1['Post']=table1['Post_m']
    
    table1['Education']=table1['Education_v'] 
    
    del table1['LoanPurpose'],table1['Post_m'],table1['ActivityType'], table1['Education_v'] 
# ----------------------------------------------------------------------------------
    table=table1[['ef_score', 'Amount', 'Education', 'MonthlyIncome', 'Post', 'cred_max_overdue_all_micro_1y_max', 'cred_sum_all_all_all_sum', 'cred_day_overdue_all_all_1y_max',
 'v0__1first_active_ilcc_all_sum', 'cred_dura_1', 'cred_sum_debt_all_all_all_mean', 'v0__12last_all_micro_all_sum', 'cred_sum_overdue_all_micro_all_max',
 'cred_sum_debt_all_micro_1y_min', 'v30__1first_all_micro_1y_mean', 're_cred_1', 'v0__1first_all_ilcc_all_mean', 'v6__12last_all_ilcc_all_mean',
 'len_num', 'cred_sum_debt_active_micro_1y_min', 'delay5_active_ilcc_all_sum', 'cred_max_overdue_active_all_1m_sum', 'all_act_cred_2',
 'cred_sum_all_micro_all_max', 'cred_sum_limit_all_all_all_mean', 'cred_sum_2', 'cred_max_overdue_active_micro_1y_mean', 'all_act_3_1_d',
 'cred_sum_debt_all_all_1y_min', 'v6__1first_active_ilcc_all_sum', 'cred_sum_33', 'cred_day_overdue_all_all_1m_min', 'cred_sum_overdue_all_micro_1y_mean',
 'partner_type_1', 'v90__1first_all_all_1y_mean', 'all_act_cred_1', 'v0__active_micro_1m_sum', 'v0__12last_active_micro_all_max', 'delay30_active_all_all_max',
 'cred_day_overdue_all_ilcc_all_mean', 'cred_sum_limit_all_all_all_sum', 'cred_max_overdue_active_ilcc_all_min', 'cred_sum_active_all_1m_min',
 'n_cr_12', 'cred_sum_active_ilcc_all_min', 'cred_sum_1', 'v0__12first_active_all_all_mean', 'cred_day_overdue_all_all_all_sum', 'v0__1first_active_ilcc_all_mean',
 'delay5_all_ilcc_all_max', 'cred_sum_all_all_1m_max', 'cred_day_overdue_all_all_1y_sum', 'v0__all_micro_all_max']]     

    table=table.astype('float')  
    if flag==1 and 1==2:        #   печать содержимого  TABLE  что идет в модель
        print('Модель 1 --------------------------------------------') 
        for index, row in table.iterrows():
            print (index, row, type(row))                   #(row['c1'], row['c2'])

        print('model.predict_proba(table):',model.predict_proba(table))

    score_1 = round(float(model.predict_proba(table)[:, 0]), 4)
                     

#   Проверяем, все-ли данные есть ?    если чего-то не хватет, то номер модели меняем на 48 - данных мало
    if flagEF == 0:
        num_model1=48  

#  ---------------------------------------------------------------------------------   XGBoost  ноябрь  -----------------
                           #   и здесь поменять !!!!!!!!!   = flag1
if 1==1:      #flagFC + flagEF == 2  and model4==1 :     #  те есть  и Финкарта и ФССП и ЕквиФакс, то эта модель работает 
 #   num_model=num_model3 
    # Loading the saved decision tree model pickle

    path2=os.path.abspath(os.path.dirname(__file__))  
    model_pkl = open(path2+'/'+'on_0_3923.pkl', 'rb')            #  проверяем ТОТ-ли файл грузится  с сохраненной моделью ??!!!!        
    model = pickle.load(model_pkl) 

    #table1['Education_v']=table1['Education'].apply(lambda x: 1 if x=='высшее' else 2)
									
    table=table1[['Amount', 'Education', 'MonthlyIncome', 'v0__active_micro_all_mean', 'cred_max_overdue_all_micro_1y_max', 'cred_annuity_all_all_1m_sum', 'v0__all_micro_all_sum',
    'cred_sum_all_all_1y_sum','cred_day_overdue_all_all_all_sum', 'cred_sum_all_micro_1m_mean', 'v0__12first_active_all_all_mean',
    'v6__12last_all_ilcc_all_mean', 'cred_day_overdue_all_all_1y_max', 'v6__all_micro_all_mean', 'cred_sum_overdue_all_micro_1y_mean',
    'FinKartaScoreV1_rateall_sum', 'cred_sum_all_micro_all_mean', 'v0__active_all_1y_mean', 'cred_sum_all_micro_all_max', 'v0__all_all_all_sum',
    'delay5_active_ilcc_all_mean', 'cred_day_overdue_all_all_1m_sum', 'v0__1first_active_ilcc_all_sum', 'FinKartaScoreV2_rateall_sum',
    'cred_day_overdue_all_all_1m_min', 'v0__12first_active_all_all_sum', 'v0__1first_all_ilcc_all_mean', 'cred_day_overdue_all_ilcc_all_mean',
    'v0__1first_all_all_1m_mean', 'cred_day_overdue_all_all_1y_sum', 'cred_max_overdue_active_all_1m_sum', 'v6__1first_active_ilcc_all_sum',
    'cred_sum_limit_all_all_all_sum', 'F_IssAmnt', 'cred_sum_all_all_1m_max', 'cred_sum_all_all_all_sum', 'v0__all_all_1m_mean',
    'delay5_all_all_1m_mean', 'cred_sum_all_micro_all_sum', 'countIssuedFor30Daysall_sum', 'cred_sum_overdue_all_all_1y_sum', 'v6__12first_all_micro_all_mean',
    'cred_sum_active_ilcc_all_min', 'cred_sum_limit_all_all_all_mean', 'v0__all_micro_all_max', 'cred_sum_debt_all_micro_1m_mean', 'v6__all_all_all_mean',
    'lastDischargeAmountall_sum', 'cred_sum_debt_all_all_all_mean', 'FinKartaScoreV2_rate', 'v0__all_micro_1y_max',
    'cred_sum_debt_all_all_all_sum','cred_max_overdue_active_ilcc_all_min']]     

    table=table.astype('float') 

    if flag==1 and 1==2:        #   печать содержимого  TABLE  что идет в модель
        print('Модель 1 --------------------------------------------') 
        for index, row in table.iterrows():
            print (index, row, type(row))                   #(row['c1'], row['c2'])

        print('model.predict_proba(table):',model.predict_proba(table))

             #   тк Модель считает вероятность Дефолта, а нам нужна Вероятность Возврата
 #   table.at[0,'Amount'] = 50000     # для теста
    score_3= round( float(model.predict_proba(table)[:, 0]), 4)
    initial_prob = score_3     #   для вывода

    if flag==1:
        print('# --------------- рассчитал скоринг для модели №63   score_3',score_3,'         table[Amount]:',table['Amount'].values[0]) 

        
# ------------------- ниже повышение суммы, если вероятность более 70 %
    if score_3 >=0.7 and table['Amount'].values[0]<=5000 :           #    -------  очень хорошие, увелчиваем сумму и выводим в базу     меняем в 3 !!!!  местах
                                            #  <=5000         те есть куда повышать.  ДО 6000 тыс - верхний лимит для онланй первичников
        score=score_3     #   текущая величина внутри цикла

        while table['Amount'].values[0] +1000 <= 6000 and score>=0.7 :    #  именно больше 3000 , тк там дальше уменьшаем сумму и пересчитываем вероятность

            a=int(table['Amount'].values[0]/100)      #  вдруг число не круглое
            a=int(a*100+1000)
            table.at[0, 'Amount'] = a

            score = round( 1- float(model.predict_proba(table)[:, 1]), 4)     #  расчет вероятности

            if flag==1:
                print('score',score,'         table[Amount]:',table['Amount'].values[0]) 

            if score>= 0.7:
                score_3=score
                Amount_3=a
                num_model3=631          

# ------------------- ниже понижение  суммы, пока вероятность не станет выше 50%, или сумма не ниже 2000 руб.
# ---           Порог меняется в 3-х  !!!!!  местах   1) вход в цикл    2)  цикл while          3) if score >0.58

# Если у клиента вероятность от 30 до 50%, то мы можем снизить ему сумму займа,

    elif score_3>=0.30 and score_3 < border and table['Amount'].values[0] >= 6000 and 1==1:             #    

        score=score_3     #   текущая величина внутри цикла                   >= 6000  те есть откуда понижать

#  Андрей 27 декабря дал ЦУ изменить механиз понижения суммы так:  с 6 тыс - до 5 тыс.,  остальное не менять

        if table['Amount'].values[0]>= 6000:
            table.at[0,'Amount'] = 5000
            num_model3=632   
            score_3= round( 1- float(model.predict_proba(table)[:, 1]), 4)     #  расчет вероятности
            Amount_3=table.at[0,'Amount']
            print("table.at[0,'Amount']",table.at[0,'Amount'], '   num_model3:',num_model3,'  Amount_3:',Amount_3)


# ------------  это выключено сейчас
        while table['Amount'].values[0]>= 3000 and score<border and 1==2:    #  именно больше 3000 , тк там дальше уменьшаем сумму и пересчитываем вероятность
            a=int(table['Amount'].values[0]/100)      #  вдруг число не круглое
            a=int(a*100-1000)
            table.at[0, 'Amount'] = a

 #            print('2 table[Amount].values[0]\n',table['Amount'].values[0])

            score = round( 1- float(model.predict_proba(table)[:, 1]), 4)     #  расчет вероятности
            if flag==1:
                print('score',score,'         table[Amount]:',table['Amount'].values[0])

            if score >= border :           #    -------  хорошие, одрбряем  и выводим в базу
                num_model3=632   
                score_3=score
                Amount_3=table.at[0,'Amount']
# ------------  это выключено сейчас


# --------------- логистическая регрессия  ==========================================================================================
if 1==1:            #     на всякий случай,чтоб можно было выключить       

    df=df.fillna(-999999)

  #  print('df.shape',df.shape)
  #  print('df.columns:',df.columns)
    

    attribs=['MonthlyIncome',]  
    for attrib in attribs:
        try :
            df[attrib]=df[attrib ].astype('float') 
        except:
            print('Ошибка:',attrib,'\n',df[attrib])

    #    В таблицу  DF добавляю столбцы со скором

    # ------------------------------   Zaim_Express_Anketa_Score_Calculation

 #   s = df['Education']
 #   for index, value in s.items():
  #      print(f"Index : {index}, Value : {value}")

    df['intercept']=0.694
    df['Education_sc']=df['Education'].apply(lambda x: 0.1735 if x=='высшее' else -0.1023)
#    print(df['Education_sc'],df['Education'])
    df['MonthlyIncome_sc']=df['MonthlyIncome'].apply(lambda x: -0.1412 if x<36863.06451613 else -0.0257 if x<52503.03225806 else 0.1319 if x<85405.87096774 else 0.4769)
#    print(df['MonthlyIncome_sc'],df['MonthlyIncome'])
    df['Gender_sc']=df['Gender'].apply(lambda x: -0.1067 if x==1 else 0.1229)
 #   print(df['Gender_sc'],df['Gender'])
    df['OrganizationPhone_sc']=df['OrganizationPhone'].apply(lambda x: -0.1099 if str(x)[:5]=='+7 (9' else 0.1554 if str(x)[:5]=='+7 (4' else 0.0154)

    df['Post_sc']=df['Post'].apply(lambda x: -0.14 if x=='рабочий' else 0.0512)

    df['ActivityType_sc']=df['ActivityType'].apply(lambda x: 0.1683 if (x=='государственная служба' or x=='наука и образование' or x=='финансы, банки, страхование, консалтинг' or x=='медицина' or x=='информационные технологии/телекоммуникации') else -0.0422)

    df['MaritalStatus_sc']=df['MaritalStatus'].apply(lambda x: 0.0917 if x==1 else -0.1107 if x==4 else -0.0478)

    df['LoanPurpose_sc']=df['LoanPurpose'].apply(lambda x: -0.1911 if x=='медицинские услуги' else 0.0238)   

    df['score_anketa']=df['intercept'] \
    + df['Education_sc'] \
    + df['MonthlyIncome_sc'] \
    + df['Gender_sc'] \
    + df['OrganizationPhone_sc'] \
    + df['Post_sc'] \
    + df['ActivityType_sc'] \
    + df['MaritalStatus_sc'] \
    + df['LoanPurpose_sc']

#    print("df['score_anketa']:",df['score_anketa'])

    # ------------------------------ Zaim_Express_BKI_Score_Calculation
    #df=df.fillna(-999999)

    df['intercept']=0.6887
    df['cred_sum_overdue_all_micro_all_mean_sc']=df['cred_sum_overdue_all_micro_all_mean'].apply(lambda x: 0.1922 if x<121.72296146 else -0.0021 if x<398.13844845 else -0.365 if x<1571.36943844 else -0.7633)
    df['cred_sum_all_micro_all_sum_sc']=df['cred_sum_all_micro_all_sum'].apply(lambda x: -0.3711 if x<71769.66451613 else -0.0403 if x<316400.31969032 else 0.1342 if x<513291.69330323 else 0.2698 if x<2510514.22964516 else 0.5866)
    df['cred_sum_active_all_all_mean_sc']=df['cred_sum_active_all_all_mean'].apply(lambda x: -0.1231 if x<10320.98836842 else 0.0332 if x<16633.36766893 else 0.0998)
    df['cred_sum_active_all_1m_min_sc']=df['cred_sum_active_all_1m_min'].apply(lambda x: -0.0833 if x<5958.54741935 else 0.2193)
    df['delay5_all_ilcc_all_sum_sc']=df['delay5_all_ilcc_all_sum'].apply(lambda x: -0.0532 if x<6 else 0.0738 if x<17.98967742 else 0.3277)
    df['v6__12last_all_micro_1y_mean_sc']=df['v6__12last_all_micro_1y_mean'].apply(lambda x: 0.0662 if x<0.31737769 else -0.1441 if x<0.78609647 else -0.3626 if x<2.58163592 else -0.6929)
    df['cred_sum_debt_all_micro_1y_min_sc']=df['cred_sum_debt_all_micro_1y_min'].apply(lambda x: -0.3768 if x<0 else 0.0356 if x<216.0444 else -0.3768)
    df['v0__12last_active_micro_all_max_sc']=df['v0__12last_active_micro_all_max'].apply(lambda x: -0.0773 if x<2 else 0.0261 if x<3 else 0.089)

    if 1==2:
        print("df['cred_sum_overdue_all_micro_all_mean']",df['cred_sum_overdue_all_micro_all_mean'],['cred_sum_overdue_all_micro_all_mean_sc'])

    df['score_bki']=df['intercept'] \
    + df['cred_sum_overdue_all_micro_all_mean_sc'] \
    + df['cred_sum_all_micro_all_sum_sc'] \
    + df['cred_sum_active_all_all_mean_sc'] \
    + df['cred_sum_active_all_1m_min_sc'] \
    + df['delay5_all_ilcc_all_sum_sc'] \
    + df['v6__12last_all_micro_1y_mean_sc'] \
    + df['cred_sum_debt_all_micro_1y_min_sc'] \
    + df['v0__12last_active_micro_all_max_sc']

 #   print("df['score_bki']:",df['score_bki'])
    # ----------------------------- Zaim_Express_Finkarta_Score_Calculation   
    #df=df.fillna(-999999)
    
    if 1==2:
        print('lastSuccessfulDischargeDateall_mean:',df['lastSuccessfulDischargeDateall_mean'].values[0])
        print('firstTransferFromMFODateall_sum:',df['firstTransferFromMFODateall_sum'].values[0])
        print('countIssuedFor90Days1_sum:',df['countIssuedFor90Days1_sum'].values[0])


    df['intercept']=0.6939
    df['lastSuccessfulDischargeDateall_mean_sc']=df['lastSuccessfulDischargeDateall_mean'].apply(lambda x: -0.2732 if x<0.60741438 else 0.3413 if x<8.25929632 else 0.1611 if x<25.63010702 else -0.1763)
    df['firstTransferFromMFODateall_sum_sc']=df['firstTransferFromMFODateall_sum'].apply(lambda x: -0.0482 if x<44.18936863 else 0.0539)
    df['countIssuedFor90Days1_sum_sc']=df['countIssuedFor90Days1_sum'].apply(lambda x: -0.098 if x<3.98322581 else 0.2375)

    df['score_finkarta']=df['intercept'] \
    + df['lastSuccessfulDischargeDateall_mean_sc'] \
    + df['firstTransferFromMFODateall_sum_sc'] \
    + df['countIssuedFor90Days1_sum_sc']

 #   print("df['score_finkarta']:",df['score_finkarta'])

    # -----------------------------------------------------        Итог скоре

    df['score_combo']=-0.4190+0.5310*df['score_anketa']+0.8472*df['score_bki']+0.2280*df['score_finkarta']
    score_combo=round(df.iloc[0]['score_combo'],6)
    score_anketa=round(df.iloc[0]['score_anketa'],6)
    score_bki=round(df.iloc[0]['score_bki'],6)
    score_finkarta=round(df.iloc[0]['score_finkarta'],6)            #        скор мой моя  
    
    score_2=round(math.exp(score_combo)/(1+math.exp(score_combo)),4)            #        вероятность моя
    
    if flag==1 and 1==2:       
        print('score_bki:', score_bki, 'score_anketa:', score_anketa,'  score_finkarta:', score_finkarta,'  score_combo:', score_combo)
# --------------------------------------------------------------------------------------------------------------


# Итоговая модель блендинга (ансамбль моделей, берем среднюю веротяность по 3 моделям)

score_res = round((score_1 + score_2 + score_3) / 3, 4)




if 1==2:
    print('{}, {}'.format(score,num_model1,prob,num_model4))             
else:                                               #  type = 1   - последовательно модели работают, type = 2 - паралельно, как здесь 
#                наша июньская                  Логистическая                                                 ноябрьская модель   
    d={'score':score_1,'num_model':num_model1,'score_2':score_2,'num_model_2':num_model2,'score_3':score_3,'num_model_3':num_model3,'type':4,'Amount': Amount_3, 'initial_prob':initial_prob, 'score_res': score_res, 'num_model_res': num_model_res}                                                  
    dj=json.dumps(d)
    print(dj)       

#type 4 - срабатывают 3 модели, на одтельные результаты внимания не обращаем, берем их среднее-арифметическое и на его основании принимаем решение - модель 65

#    num_model1=67            #  Модель со скором Эквифакса
                         #   48  пишется, когда ничего не сработало и тогда веротяность = 0.01
#    num_model4=62            #   Логистическая регрессия

#    num_model3=63            #   "XGBoost   модель Наташи от ноября 2021"




