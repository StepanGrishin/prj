#!/usr/bin/env python
# coding: utf-8

#                            Скрипт для ОН-лайн для 5-кратников, используются базы: Анкета + БКИ  
#                            Сохраненная модель в файле:  on_5_4534.pkl – с данными Эквифакса       
#                                                         on_5_4502.pkl – без данных Эквифакса                                                        
#         Затем запускается модели логистической регресии , номер у которой 61

import pickle
import pandas as pd
import numpy as np
import xgboost
from xgboost import XGBClassifier
#from sklearn.preprocessing import LabelEncoder
import json
import sys
import os
import math
import re

##################  Задание номера моделей  num_model   для записи в базу

#без данных  Эквифакса
num_model_1=150    #   150 - ничего не меняю,   151 - повышаю сумма,   152  понижаю сумму 

# с данными экифакса
num_model_2=160    #   160 - ничего не меняю,   161 - повышаю сумма,   162  понижаю сумму 

#    15 модель -   для ОФФ-лайна

 
model1=1        # 0 - Модель выключена,  1 - модель включена    
model2=1	# 0 - Модель выключена,  1 - модель включена
model3=1	# 0 - Модель выключена,  1 - модель включена
model4=1	# 0 - Модель выключена,  1 - модель включена    

     #   и внизу поменять !!!!!!!!!
#flag1=2        #  те сумма флагов ==2     если 3 то включена модель по трем источникам, а если например 4 - то выключена
##########################################
flag=0        #    flag=1  Печатать комменты,   flag=0  - не печатать 
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


#df=table.T.copy()        #  для логистической регрессии
#print('df.shape',df.shape)

#table.replace(to_replace='null', value=np.nan, inplace=True) 

if flag==1 and 1==2:                #  1  - печатать комменты
    Fin = open (sys_argv) 
    print(Fin.read())
    Fin.close()
    print('\n')

table=table.T  


#   Служебная информация для моделей ---------------------

ServiceData=table['ServiceData'].values[0]  
RequestAmount= ServiceData.get("RequestAmount")

LoanNumbers= ServiceData.get("LoanNumbers")
LoanNumbers_off = LoanNumbers.get("Offline")    #   Вытаскиваю  Порог конкретной модели ДЛЯ изменения суммы

#   LoanNumbers - словарь где указаны число ранее взятых займов в ОФФ и ОН-лайне
upper_limit=15000    #   верхний лимит при увеличении суммы

if LoanNumbers_off>=1:
    upper_limit=20000
    
    
    
upper_limit_160 = 30000

if flag==1 and 1==1:
    print ('ServiceData:', ServiceData)
    print ('RequestAmount:', RequestAmount)

    print ('LoanNumbers_off',LoanNumbers_off)
 


#   Borders - словарь где все Номера моделей и все значения  границ  - уровней Одобрения
if 1==2:
    Borders=table['Borders'].values[0]        

    border_1 = Borders.get(str(num_model_1))/100    #   Вытаскиваю  Порог конкретной модели ДЛЯ изменения суммы
    border_2 = Borders.get(str(num_model_2))/100    #   Вытаскиваю  Порог конкретной модели ДЛЯ изменения суммы

if flag==1 and 1==2:
    print('border:' , border_1, ' of', num_model_1)


#df=table.copy()        #  для логистической регрессии
#df=table.T.copy()        #  для логистической регрессии
#table1=table.copy()       #  для  новой модели

if flag==1 and 1==2:                #  1  - печатать комменты    
    print('ТИпы данных:\n',table.shape, table.info())     

#flagFC=todos['flagFC']
#flagSP=todos['flagSP']
flagEF=todos['flagEF']

if flag==1:
    print('flagEF',flagEF)

Amount_1=''                    #    измененная сумма , или вверх , или вниз

#  ---------------- с данными экифакса  -----   num_model_2=160    #   160 - ничего не меняю,   161 - повышаю сумма,   162  понижаю сумму 
                           #   и здесь поменять !!!!!!!!!   = flag1
if flagEF == 1  and  model1==1 :     #  те есть  ЭквиФакс, то эта модель работает 
 #   num_model=num_model1 
    # Loading the saved decision tree model pickle
    if 1==1:     #  боевой вариант    берем файл с моделью с сервера
        path2=os.path.abspath(os.path.dirname(__file__))           #  где лежит исполняемый срипт и рядом сохраненная модель
        model_pkl = open(path2+'/'+'on_5_4534.pkl', 'rb')            #  проверяем ТОТ-ли файл грузится  с сохраненной моделью ??!!!!
    else :       #  тестовый вариант   берем из файла на компе, когда модель запускается на компе
        path2='c:\\tmp\\'
        model_pkl = open(path2+'on_5_4534.pkl', 'rb')                #  проверяем ТОТ-ли файл грузится  с сохраненной моделью ??!!!! 

    model = pickle.load(model_pkl) 

    table=table[['Fact_duration_loans_max', 'Amount', 's_sum', 'v6__1first_all_all_all_mean', 'cred_sum_all_ilcc_all_sum',
		'Education', 'v6__all_all_1y_mean', 'v0__1first_all_all_all_mean', 'cred_sum_active_all_1y_sum',
		'cred_sum_overdue_all_micro_all_sum', 'cred_sum_overdue_all_ilcc_all_mean', 'v0__12last_active_ilcc_all_mean',
		'cred_sum_overdue_all_micro_all_mean', 'delay30_all_micro_all_mean', 'age', 'cred_sum_all_micro_all_mean',
		'v0__1first_active_all_all_sum', 'cred_sum_all_all_all_max', 'v0__all_all_all_sum', 'v0__12first_all_ilcc_all_mean',
		'cred_day_overdue_all_all_all_sum', 'cred_sum_all_all_all_sum', 'v0__12first_active_ilcc_all_mean',
		'cred_sum_all_micro_all_sum', 'cred_sum_active_all_1y_mean', 'v0__active_all_all_mean', 'MonthlyIncome',
		'cred_sum_active_micro_all_mean', 'cred_sum_active_micro_all_min', 'cred_day_overdue_all_all_1y_max',
		'cred_day_overdue_all_all_1y_sum', 'cred_day_overdue_all_all_1m_max', 'cred_max_overdue_all_micro_all_mean',
		'cred_sum_all_micro_all_max', 'cred_sum_all_micro_1m_max', 'cred_sum_debt_all_ilcc_all_sum',
		'v0__1first_all_ilcc_all_sum', 'cred_day_overdue_all_ilcc_all_sum', 'cred_sum_all_all_1m_sum',
		'cred_sum_debt_all_micro_all_max', 'cred_sum_all_all_1y_min', 'cred_sum_overdue_all_all_1y_sum',
		'cred_day_overdue_all_all_1y_mean', 'N_Loans_befo', 'cred_sum_all_all_1y_sum', 'cred_sum_limit_all_all_all_sum',
		'cred_day_overdue_all_micro_all_mean', 'cred_sum_active_ilcc_all_min', 'cred_sum_debt_all_all_all_sum',
		'delay5_all_all_all_max', 'cred_sum_overdue_all_micro_1y_mean', 
		 'cred_day_overdue_all_micro_all_max']]    
        
 #   table.replace(to_replace='null', value=np.nan, inplace=True)     
        
#обрабатываем категориальные переменные:
    others= ['высшее', 'среднее специальное', 'неполное высшее', 'среднее', 'неполное среднее']

    table['Education']=table['Education'][0] if table['Education'].isin(others)[0] else 'others'

    table['Education']=table['Education'].map({'высшее': 0.195, 'среднее специальное': 0.255, 'неполное высшее': 0.219
                                                 , 'среднее': 0.256, 'неполное среднее': 0.228, 'others': 0.154})  
    
    table=table.astype(float)
   
    if 1==2:
        table.at[0,'s_sum'] = 0   # np.nan  #0     #  для отладки
        
    if 1==2:
        for index, row1 in table.iterrows():
            print (row1)
       
    initial_prob=round( 1- float(model.predict_proba(table)[:, 1]), 4)   #  расчет вероятности
               #  запоминаю начальные значения тк вдруг на первой итерации уйду ниже 85 %
        
    initial_Amount= table['Amount'].values[0]      
    score_1=initial_prob       
                    
 #   Amount_1=table['Amount'].values[0]


# Повышение: если вероятность больше 80%, то повышаем сумму в рамках лимитов, но так, чтобы вероятность не упала ниже 80%.
#   (15000 руб. максимум для онлайн, и 30000 руб. – если клиент подтвердил свою личность в офисе выдачи)    
  
#сделаем 161  - для повышения суммы, а 162 - для понижения.     160  - без изменения


#  Поэтому таким клиентам можно сразу 15 предлагать, если вероятность больше 75%.    
 
    score = initial_prob
    Amount_initial=RequestAmount
    num_model_1=160

    if flag==1:
        print("score , RequestAmount,num_model_1" )
        print(score , RequestAmount,num_model_1 )


    if score >0.75:           #    -------  очень хорошие, увелчиваем сумму и выводим в базу  

        if flag==1:
            print('Amount_initial:',Amount_initial, table['Amount'].values[0], '  upper_limit:',upper_limit_160)  
      
        if  RequestAmount<upper_limit_160:           #   обычно 15 тыс,  но когда был челвоек в офисе , то 20000
 #           print(RequestAmount, table['Amount'].values[0], math.floor(table['Amount'].values[0]*2/1000)) 
            table.at[0, 'Amount'] =math.floor(RequestAmount*2/1000)*1000        #  АНдрей сказал увеличивать сумму не более чем в 2 раза

   
            if table['Amount'].values[0]>upper_limit_160:
                table.at[0, 'Amount'] = upper_limit_160
           
            Amount_1=table['Amount'].values[0]

 #           print(Amount_1)
            if Amount_1> RequestAmount:
                num_model_1=161

        if flag==1:
            print( "Amount_1,  RequestAmount,  num_model_1")  
            print( Amount_1,  RequestAmount,  num_model_1)  
            
    elif  0.50<score and score<=0.55:              #  >50<=55	25,0%
        table.at[0, 'Amount'] = math.floor(RequestAmount*0.75/1000)*1000
        num_model_1=162
        Amount_1=table.at[0, 'Amount'] 
        
    elif  0.45<score and score<=0.50:              #  >45<=50	30,0%
        table.at[0, 'Amount'] = math.floor(RequestAmount*0.70/1000)*1000
        num_model_1=162
        Amount_1=table.at[0, 'Amount'] 
        
    elif  0.40<score and score<=0.45:              #    >40<=45	35,0%
        table.at[0, 'Amount'] = math.floor(RequestAmount*0.65/1000)*1000 
        num_model_1=162        
        Amount_1=table.at[0, 'Amount'] 
        
    elif  0.35<score and score<=0.40:              #    >35<=40	40,0%
        table.at[0, 'Amount'] = math.floor(RequestAmount*0.60/1000)*1000
        num_model_1=162
        Amount_1=table.at[0, 'Amount'] 
        
    elif  0.30<score and score<=0.35:              #    >30<=35	40,0%
        table.at[0, 'Amount'] = math.floor(RequestAmount*0.60/1000)*1000       
        num_model_1=162
        Amount_1=table.at[0, 'Amount'] 
        
    elif  0.20<score and score<=0.30:              #    >>20<=30	45,0%

        table.at[0, 'Amount'] = math.floor(RequestAmount*0.55/1000)*1000  
        num_model_1=162
        Amount_1=table.at[0, 'Amount'] 
        
    elif  0.10<score and score<=0.2:              #    >>10<=20	50,0%
        table.at[0, 'Amount'] = math.floor(RequestAmount*0.50/1000)*1000          
        num_model_1=162
        Amount_1=table.at[0, 'Amount'] 
        
    elif  score<=0.1:              #    >>10<=20	50,0%
        table.at[0, 'Amount'] = math.floor(RequestAmount*0.40/1000)*1000 
        num_model_1=162
        Amount_1=table.at[0, 'Amount'] 
        
    if table['Amount'].values[0]<2000:
        table.at[0, 'Amount']=2000
        Amount_1=table.at[0, 'Amount'] 
                       
# -------------- случай, когда нет ответа от Эквифакса те сумма до 10 тыс. num_model_1=150    
#   150 - ничего не меняю,   151 - повышаю сумма,   152  понижаю сумму 

else:     

    path2=os.path.abspath(os.path.dirname(__file__))  
    model_pkl = open(path2+'/'+'on_5_4502.pkl', 'rb')            #  проверяем ТОТ-ли файл грузится  с сохраненной моделью ??!!!!        
    model = pickle.load(model_pkl) 

    def clean_job_exp(x):
		
        if type(x) == 'int' or type(x) == 'float':
            return x
	
        elif (('более') in x.lower() or ('меньше') in x.lower() or ('менее') in x.lower() or ('больше') in x.lower()) and ('мес') not in x.lower().split(' ')[2]:
            return float(x.split(' ')[1])
	
        elif (('год') in x.lower() or ('лет') in x.lower()) and ('-') not in x.lower():
            return float(re.findall('\d+', x)[0])
	
        elif (('более') in x.lower() or ('меньше') in x.lower() or ('менее') in x.lower() or ('больше') in x.lower()) and ('мес') in x.lower().split(' ')[2]:
            return 1
	
        elif ('мес') in x.lower():
            return 1
	
        elif ('не работал') in x.lower() or x == 'None':
            return 0
	
        elif (('год') in x.lower() or ('лет') in x.lower()) and ('-') in x.lower():
            return (float(x.lower().split(' ')[0].split('-')[0]) + float(x.lower().split(' ')[0].split('-')[1])) / 2
        else: 
            return x
     
 #   table.replace(to_replace='null', value=np.nan, inplace=True)     
        
#    OrganizationPhone_v:
    table['OrganizationPhone'] = table['OrganizationPhone'].astype('str')
    table['OrganizationPhone_v'] = table.apply(lambda x: 1 if '+7 (9' in x['OrganizationPhone'] else 2, axis = 1)
    table['OrganizationPhone_v'] = table.apply(lambda x: 3 if '+7 (4' in x['OrganizationPhone'] else x['OrganizationPhone_v'], axis = 1)     
               
#обрабатываем категориальные переменные:
    others= ['высшее', 'среднее специальное', 'неполное высшее', 'среднее', 'неполное среднее']

    table['Education']=table['Education'][0] if table['Education'].isin(others)[0] else 'others'

    table['Education']=table['Education'].map({'высшее': 0.195, 'среднее специальное': 0.255, 'неполное высшее': 0.219
                                                 , 'среднее': 0.256, 'неполное среднее': 0.228, 'others': 0.154})  
 
    if 1==2:
        print("table['LastJobExpirience'][0]",table['LastJobExpirience'][0])
        ff = clean_job_exp(table['LastJobExpirience'][0])
        print('ff',ff)


    table['LastJobExpirience'] = table['LastJobExpirience'].astype('str').apply(clean_job_exp)
    table['LastJobExpirience'] = table['LastJobExpirience'].astype('float')
 #   print('LastJobExpirience',table['LastJobExpirience'][0])


    non_others=['nan','Менеджер', 'Инженер', 'Продавец', 'Оператор', 'Водитель', 'Менеджер ', 'Администратор', 'Специалист', 
                'Администратор ', 'Менеджер по продажам', 'Бухгалтер', 'Кладовщик', 'Водитель ', 'Ведущий специалист', 'Мастер',
                'Начальник отдела', 'Специалист ', 'Слесарь', 'Кассир', 'Главный специалист', 'Менеджер по продажам ', 
                'Продавец ', 'Инженер ', 'Ведущий специалист ', 'Повар', 'Учитель', 'Воспитатель', 'Экономист', 'Оператор ', 
                'менеджер', 'Воспитатель ', 'Кладовщик ', 'Курьер', 'Продавец кассир', 'Бухгалтер ', 'Юрист']
    
    table['PostSpecification']=table['PostSpecification'].fillna('nan')
 #   print('1 table[PostSpecification]', table['PostSpecification'][0])

    table['PostSpecification']=table['PostSpecification'][0] if table['PostSpecification'].isin(non_others)[0] else 'others'

 #   print('2 table[PostSpecification]', table['PostSpecification'][0])
    
    
    table['PostSpecification']=table['PostSpecification'].map({'others': 0.221, 'nan': 0.205, 'Менеджер': 0.223,
                            'Инженер': 0.204, 'Продавец': 0.276, 'Оператор': 0.249, 'Водитель': 0.245, 'Менеджер ': 0.222, 
                            'Администратор': 0.209, 'Специалист': 0.244, 'Администратор ': 0.252, 'Менеджер по продажам': 0.188,
                            'Бухгалтер': 0.245, 'Кладовщик': 0.257, 'Водитель ': 0.214, 'Ведущий специалист': 0.165, 
                            'Мастер': 0.204, 'Начальник отдела': 0.167, 'Специалист ': 0.222, 'Слесарь': 0.126, 'Кассир': 0.27,
                            'Главный специалист': 0.207, 'Менеджер по продажам ': 0.213, 'Продавец ': 0.296, 'Инженер ': 0.223,
                            'Ведущий специалист ': 0.206, 'Повар': 0.301, 'Учитель': 0.256, 'Воспитатель': 0.31, 
                            'Экономист': 0.135, 'Оператор ': 0.21, 'менеджер': 0.172, 'Воспитатель ': 0.187, 
                            'Кладовщик ': 0.248, 'Курьер': 0.239, 'Продавец кассир': 0.279, 'Бухгалтер ': 0.237, 'Юрист': 0.208})          
        
    non_others=['непредвиденные расходы', 'ремонт дома или автомобиля', 'другое', 'ежедневные расходы', 'медицинские услуги',
                'подарки', 'покупка бытовой техники и электроники', 'образование', 'путешествия, отдых', 'бизнес']  

    table['LoanPurpose']=table['LoanPurpose'][0] if table['LoanPurpose'].isin(non_others)[0] else 'others'

    table['LoanPurpose']=table['LoanPurpose'].map({'непредвиденные расходы': 0.218, 'ремонт дома или автомобиля': 0.23
                                                    , 'другое': 0.227, 'ежедневные расходы': 0.211, 'медицинские услуги': 0.236
                                                    , 'подарки': 0.196, 'покупка бытовой техники и электроники': 0.222
                                                    , 'образование': 0.194, 'путешествия, отдых': 0.19, 'бизнес': 0.241
                                                    , 'others': 0.139})          
            
    
    non_others=['служащий / специалист', 'рабочий', 'начальник / руководитель', 'другое', 'руководитель компании', 
                'водитель', 'медсестра/медбрат', 'главный бухгалтер', 'офицер', 'рядовой', 'врач']    
        
       
 #   table['Post']=table['Post'][0] if table['Post'].isin(others)[0] else 'others' 
 #   print('1 table[Post][0]', table['Post'][0])
    table['Post']= table['Post'][0] if table['Post'].isin(non_others)[0] else  'others'
    
 #   print('2 table[Post][0]', table['Post'][0])    
    
    table['Post']=table['Post'].map({'служащий / специалист': 0.222, 'рабочий': 0.245, 
            'начальник / руководитель': 0.183, 'другое': 0.217, 'руководитель компании': 0.195, 'водитель': 0.224, 
            'медсестра/медбрат': 0.238, 'главный бухгалтер': 0.186, 'офицер': 0.219, 'рядовой': 0.265, 'others': 0.165, 
            'врач': 0.208})      
    
 #   print('3 table[Post][0]', table['Post'][0])    
 
 
    table=table[['Amount', 'day_from_closed', 'AmountOfMonthlyLoanCosts', 'Education', 'Dola_60_90', 'Amount_min',
                 'N_record_IP', 'Fact_duration_loans_max', 'PostSpecification', 'max_days_delay_sum', 'MonthlyIncome',
                 'EMailVerified', 'day_from_closed_min', 'LoanPurpose', 's_sum', 'Fact_duration_loans_mean',
                 'day_from_closed_sum', 'N_payments_sum', 'OrganizationPhone_v', 'N_payments_mean', 
                 'plan_Duration_max', 'Fact_duration_loans_sum', 'cred_date_last_2', 'LastJobExpirience', 'max_days_delay_min',
                 'Post', 'Gender', 'day_from_closed_mean', 'Amount_mean', 'Fact_duration_loans_min', 'plan_Duration_mean',
                 'N_Loans_befo', 'age']]

    table=table.astype(float)

    
    if 1==2:    # np.nan  #0     #  для отладки
        table.at[0,'s_sum'] = 0   # np.nan  #0     #  для отладки
     
    if 1==2:
        for index, row1 in table.iterrows():
            print (row1)
       
    initial_prob=round( 1- float(model.predict_proba(table)[:, 1]), 4)   #  расчет вероятности
               #  запоминаю начальные значения тк вдруг на первой итерации уйду ниже 85 %
        
    Amount_1=table['Amount'].values[0]

    if flag==1:
        print('initial_prob:', initial_prob, ' Amount_1:',Amount_1 )

# Повышение: если вероятность больше 80%, то повышаем сумму в рамках лимитов, но так, чтобы вероятность не упала ниже 80%.
#   (15000 руб. максимум для онлайн, и 20000 руб. – если клиент подтвердил свою личность в офисе выдачи)    
  
#сделаем 151  - для повышения суммы, а 152 - для понижения.     150  - без изменения
    score_1=initial_prob
    score=initial_prob   
    max_Amount =  math.floor(RequestAmount*2/1000)*1000    
    if max_Amount>upper_limit:  
        max_Amount=upper_limit

    if flag==1:
        print("table.at[0, 'Amount'], RequestAmount, Amount_1, score   upper_limit" )
        print(table.at[0, 'Amount'], RequestAmount, Amount_1, score, upper_limit )



    if score >0.80:           #    -------  очень хорошие, увелчиваем сумму и выводим в базу  

        table.at[0, 'Amount']=RequestAmount

        if flag==1:
            print("table.at[0, 'Amount'], RequestAmount, score   upper_limit" )
            print(table.at[0, 'Amount'], RequestAmount, score, upper_limit )

        while table['Amount'].values[0] +1000 <= max_Amount and score>0.80:    #  

            table.at[0, 'Amount'] = table['Amount'].values[0]+1000       

            score = round( 1- float(model.predict_proba(table)[:, 1]), 4)     #  расчет вероятности

            if flag==1:
                print('score',score,'         table[Amount]:',table['Amount'].values[0]) 

            if score>=0.80:
                score_1=score
                Amount_1=math.floor(table['Amount'].values[0]/1000)*1000  
                num_model_1=151
            if flag==1:
                print("score_1, Amount_1, num_model_1" )
                print(score_1, Amount_1, num_model_1 )

    #сделаем 151  - для повышения суммы, а 152 - для понижения.

    #  1.	Понижение: если вероятность меньше или равна 50%, то понижаем сумму до тех пор, пока вероятность не станет 
    #                больше 50% или сумма меньше 2000 руб. 

    elif score <=0.50   and   1==1:          #    меняю сумму займа, чтоб изменилась вероятность,   "1==2" - выключение уменьшения суммы займа

        while table['Amount'].values[0]>= 3000 and score<=0.50:    #  именно больше 3000 , тк там дальше уменьшаем сумму и пересчитываем вероятность

            table.at[0, 'Amount'] = table['Amount'] - 1000

            score = round( 1- float(model.predict_proba(table)[:, 1]), 4)     #  расчет вероятности

            if flag==1:
                print('score',score,'         table[Amount]:',table['Amount'].values[0])

            if score >0.50:           #    -------  хорошие, одрбряем  и выводим в базу
                score_1=score
                Amount_1=math.floor(table['Amount'].values[0]/1000)*1000  
                num_model_1=152    

#   --------------------------------------------------------------------------------------------------------------

if 1==2:
    print('{}, {}'.format(score,num_model1,prob,num_model4))             
else:                                               #  type = 1   - последовательно модели работают, type = 2 - паралельно, как здесь 
#                наша июньская                  Логистическая                                                 ноябрьская модель   
    d={'score':score_1,'num_model':num_model_1,'type':2,'Amount': Amount_1, 'initial_prob':initial_prob}                                                  
    dj=json.dumps(d)
    print(dj)       



##################  Задание номера моделей  num_model   для записи в базу

#без данных  Эквифакса
num_model_1=150    #   150 - ничего не меняю,   151 - повышаю сумма,   152  понижаю сумму 

# с данными экифакса
num_model_2=160    #   160 - ничего не меняю,   161 - повышаю сумма,   162  понижаю сумму 

#    15 модель -   для ОФФ-лайна



