#----------------------------------------------------------------------------------------------------------------------------------------------------
# This File has four different input formats : 
# (1). Form/Application Input Format - POSTMAN - TEXT OUTPUT SINGLE ROW : You must provide form input format in POSTMAN under Body-->Form Data(key-value-edit)
# (2). FORM Input Format - JSON OUTPUT-FORMAT -POSTMAN - MULTIPLE ROWS - You must provide filters form input format in POSTMAN under Body-->Form Data this has 3 scenarios
                 #-----------------------------------------------------------------------------------------------------------------------------------
##              (2a).Scenario-1 :This Scenartio will Updates the Payposition If Payposition is greater than Threshold value then it will add Payposition 
                #  with Above Threshold else it will Add Payposition with below Threshold
                 # You must provide six filters in form format (POSTMAN)under Body-->Form Data for this scenario of which grade and sub_bu should be list of multiple filters 
                 # gender:Male,unified_grade:['A5','B1'] sub_bu:["BPS Oracle","BPS Big Data"] Threshold:0.89,Above Threshold:0.2,Below Threshold:0.5
                 #-----------------------------------------------------------------------------------------------------------------------------------
                #(2b).Scenario-2 :This Scenartio will promotes the employee to next grade and update months_since_last_promotion =0 
               # You must provide three filters in form format (POSTMAN)under Body-->Form Data for this scenario of which grade and sub_bu should be list of multiple filters 
                # gender:Male,unified_grade:['A5','B1'] sub_bu:["BPS Oracle","BPS Big Data"]
                 #-----------------------------------------------------------------------------------------------------------------------------------
                #(2c).Scenario-3 :This Scenartio will just update education_score from "53" to "81" and make predictions on top of it
               # You must provide three filters in form format (POSTMAN)under Body-->Form Data for this scenario of which grade and sub_bu should be list of multiple filters 
                # gender:Male,unified_grade:['A5','B1'] sub_bu:["BPS Oracle","BPS Big Data"]
                 #-----------------------------------------------------------------------------------------------------------------------------------
# (3). JSON Input Format - JSON OUTPUT FORMAT-POSTMAN - MULTIPLE ROWS - You must provide filters JSON input format in POSTMAN under Body-->raw this has 3 scenarios
                 #-----------------------------------------------------------------------------------------------------------------------------------
##              (3a).Scenario-1 :This Scenartio will Updates the Payposition If Payposition is greater than Threshold value then it will add Payposition 
                #  with Above Threshold else it will Add Payposition with below Threshold
                 # You must provide six filters in json format (POSTMAN)under Body-->raw for this scenario of which grade and sub_bu should be list of multiple filters 
                # [{"gender": "Female","unified_grade": ["B1","A5"],"sub_bu":["BPS Oracle","BPS Big Data"],"Threshold":0.89,Above Threshold:0.2,Below Threshold:0.5}]
                 #-----------------------------------------------------------------------------------------------------------------------------------
                #(3b).Scenario-2 :This Scenartio will promotes the employee to next grade and update months_since_last_promotion =0 
               # You must provide three filters  json format (POSTMAN)under Body-->Form Data for this scenario of which grade and sub_bu should be list of multiple filters 
                # [{"gender": "Female","unified_grade": ["B1","A5"],"sub_bu":["BPS Oracle","BPS Big Data"]}]
                 #-----------------------------------------------------------------------------------------------------------------------------------
                #(3c).Scenario-3 :This Scenartio will just update education_score from "53" to "81" and make predictions on top of it
               # You must provide three filters  json format (POSTMAN)under Body-->Form Data for this scenario of which grade and sub_bu should be list of multiple filters 
                # [{"gender": "Female","unified_grade": ["B1","A5"],"sub_bu":["BPS Oracle","BPS Big Data"]}]

# (4). JSON Input Format - JSON OUTPUT-POSTMAN - MULTIPLE ROWS - OUTPUT will be predicted with probabilities and status for No.of rows provided
#-------------------------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
import psycopg2
import json
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
conn = psycopg2.connect(database="ds_db",user="postgres", password="HRA@2021", host="10.246.90.166")
conn.autocommit = True
cur= conn.cursor()
data = pd.read_sql('select * from Classification_Output',conn)
##------------------------------------------------------------------------------------------------------------
                                 ## (1). Form/Application Input Format - POSTMAN - SINGLE ROW 
##------------------------------------------------------------------------------------------------------------
## This API will Get the Predictions from Catboost Model based on form Input Data from POSTMAN for Single Row
## Output will be return in Text format with Attrition Probability
##------------------------------------------------------------------------------------------------------------

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    prob = model.predict_proba(final_features)
    term_output='{0:.{1}f}'.format(prob[0][1], 2)
    retent_output='{0:.{1}f}'.format(prob[0][0], 2)

    if term_output>str(0.5):
        prediction_text = "Attrition Probability {}:".format(term_output)
        return prediction_text 
    else:
        prediction_text = "Retention Probability {}:".format(retent_output)
        return prediction_text 
##-------------------------------------------------------------------------------------------------------------------------------------------------------
                                                      # (2). FORM Input Format - JSON OUTPUT-FORMAT - POSTMAN
##-------------------------------------------------------------------------------------------------------------------------------------------------------
## (2a). Scenario - 1 : This Scenartio will Updates the Payposition based on Threshold values Input from Postman - for Multiple Rows-Multiple grade and sub_bu
## If Payposition is greater than Threshold value then it will add Payposition with Above Threshold else it will Add Payposition with below Threshold
# You must provide six filters (POSTMAN)under Body-->Form Data for this scenario of which grade and sub_bu should be list of multiple filters 
# gender:Male,unified_grade:['A5','B1'] sub_bu:["BPS Oracle","BPS Big Data"] Threshold:0.89,Above Threshold:0.2,Below Threshold:0.5
##-------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/scenario1',methods=['GET','POST'])
def scenario1():
    global df1,store_data,grade,subbu		
    store_data = pd.DataFrame()
    df = data.copy()
    df1 = df.loc[(df['employee_status']==0) & (df['pred']==1)]
    if request.method=='POST':  
        gdr = request.form.get('gender') 
        thrs = request.form.get('Threshold')
        AT = request.form.get('Above Threshold')
        BT = request.form.get('Below Threshold') 
##------------------------------------------------------------------------------------------------------------------------------------
#The below line will return single list of values from Postman whose length is '1' (output:["A5,B1,B2"] ["BPS Oracle,BPS Big Data"]     
##------------------------------------------------------------------------------------------------------------------------------------  
        grade = request.form.getlist('unified_grade')
        subbu = request.form.getlist('sub_bu')
##--------------------------------------------------------------------------------------------------------------------------------------------------------
# Formatting the grade variable by removing [ , ] , ' this will give "A5,B1,B2" and "BPS Oracle,BPS Big Data" then finally split by using comma separator 
  ## which will return output in list format like ['A5,'B1','B2'] whose length would be '3' for grade and '2' for sub_bu.
##-------------------------------------------------------------------------------------------------------------------------------------------------------   
        grade_str =str(grade)
        grade_str=grade_str.strip('[')
        grade_str=grade_str.strip(']')
        grade_str=grade_str.strip("'")
        grade_str=grade_str.split(",")
        subbu_str =str(subbu)
        subbu_str =subbu_str.strip('[')
        subbu_str =subbu_str.strip(']')
        subbu_str= subbu_str.strip("'")
        subbu_str=subbu_str.split(",")
##-------------------------------------------------------------------------------------------------------------------------------
# Inorder To Apply filters on multiple values of grade with list datatype then we must use .isna(grade_str) & .isna(subbu_str)
##-------------------------------------------------------------------------------------------------------------------------------
        df1 = df1[df1['unified_grade'].isin(grade_str)]
        df1 = df1[df1['sub_bu'].isin(subbu_str)]   
##---------------------------------------------------------------------------------------------------------------------------------             
        df1 = df1[df1['gender']==gdr]  
        df1 = df1[['global_emp_id','gender','unified_grade','city','sub_bu','ijp_90_days','tenure','bench_ageing','marital_status',
            'education_score','leaves','months_since_last_promotion','payposition_2021','tenure_in_capgemini_yr',
            'average_rating','rating_diff']]    
        df1['payposition_2021'] = df1['payposition_2021'].apply(lambda x: x+float(AT) if x >= float(thrs) else x+float(BT))            
        global pred_value1,attrite_count,retain_count
        # here model is nothing but Catboost  model(model) which I have created above.
        attrite_count,retain_count = 0,0
        for index,row in df1.iterrows():
            row = pd.DataFrame(row)
            row_new = row.T
            store_data = store_data.append(row_new)
            row_new.drop('global_emp_id',inplace=True,axis=1)
            pred_value1 = model.predict(row_new)
            global ab
            ab = model.predict_proba(row_new)        
            store_data.at[index,'Termination_probability'] = ab[0][1]
            store_data.at[index,'Retention_probability'] = ab[0][0]
            if pred_value1 == 0:
                retain_count = retain_count+1
            else:
                attrite_count = attrite_count+1
        text =  "Employee Attrition Count before changing Payposition was :: {}".format(df1.shape[0])
        text1 = 'Employee Attrition Count After changing Payposition is :: {}'.format(attrite_count)
        text2 = 'Employee Retention Count After changing Payposition is :: {}'.format(retain_count)    
        scenario = "Scenario 1"
        query = store_data.to_json(orient='records')
        json_data= json.loads(query)
        return jsonify(json_data)
##-------------------------------------------------------------------------------------------------------------------------------------------------------
                                                      # (2). FORM Input Format - JSON OUTPUT-FORMAT - POSTMAN
##-------------------------------------------------------------------------------------------------------------------------------------------------------
## (2b).Scenario - 2 : This Scenario will Promotes the employee to next grade and updates months_since_last_promotion to "0" - for Multiple Rows
## Filters on Multiple grades and sub_bu's
# You must provide three filters (POSTMAN)under Body-->Form Data for this scenario of which grade and sub_bu should be list of multiple filters 
# gender:Male,unified_grade:['A5','B1'] sub_bu:["BPS Oracle","BPS Big Data"]
##------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/scenario2',methods=['GET','POST'])
def scenario2():
    global df1,store_data
    store_data = pd.DataFrame()
    df = data.copy()
    df1 = df.loc[(df['employee_status']==0) & (df['pred']==1)]
    if request.method=='POST':  
        gdr = request.form.get('gender')
##------------------------------------------------------------------------------------------------------------------------------------
#The below line will return single list of values from Postman whose length is '1' (output:["A5,B1,B2"] ["BPS Oracle,BPS Big Data"]     
##------------------------------------------------------------------------------------------------------------------------------------  
        grade = request.form.getlist('unified_grade')
        subbu = request.form.getlist('sub_bu')
##--------------------------------------------------------------------------------------------------------------------------------------------------------
# Formatting the grade variable by removing [ , ] , ' this will give "A5,B1,B2" and "BPS Oracle,BPS Big Data" then finally split by using comma separator 
  ## which will return output in list format like ['A5,'B1','B2'] whose length would be '3' for grade and '2' for sub_bu.
##-------------------------------------------------------------------------------------------------------------------------------------------------------   
        grade_str =str(grade)
        grade_str=grade_str.strip('[')
        grade_str=grade_str.strip(']')
        grade_str=grade_str.strip("'")
        grade_str=grade_str.split(",")
        subbu_str =str(subbu)
        subbu_str =subbu_str.strip('[')
        subbu_str =subbu_str.strip(']')
        subbu_str= subbu_str.strip("'")
        subbu_str=subbu_str.split(",")
##-------------------------------------------------------------------------------------------------------------------------------
# Inorder To Apply filters on multiple values of grade with list datatype then we must use .isna(grade_str) & .isna(subbu_str)
##-------------------------------------------------------------------------------------------------------------------------------
        df1 = df1[df1['unified_grade'].isin(grade_str)]
        df1 = df1[df1['sub_bu'].isin(subbu_str)]                    
##---------------------------------------------------------------------------------------------------------------------------------  
        df1 = df1[df1['gender']==gdr]            
        df1 = df1[['global_emp_id','gender','unified_grade','city','sub_bu','ijp_90_days','tenure','bench_ageing','marital_status',
            'education_score','leaves','months_since_last_promotion','payposition_2021','tenure_in_capgemini_yr',
            'average_rating','rating_diff']]                 
        global pred_value1,attrite_count,retain_count
        # here model is nothing but Catboost  model(model) which I have created above.
        attrite_count,retain_count = 0,0
        for index,row in df1.iterrows():
            row = pd.DataFrame(row)
            row_new = row.T
            if (row_new['unified_grade'] == 'A3').all():
                row_new.at[index,'unified_grade']='A4'
            elif (row_new['unified_grade'] == 'A4').all():
                row_new.at[index,'unified_grade']='A5'  
            elif (row_new['unified_grade'] == 'A5').all():
                row_new.at[index,'unified_grade']='B1'
            elif (row_new['unified_grade'] == 'B1').all():
                row_new.at[index,'unified_grade']='B2'
            elif (row_new['unified_grade'] == 'B2').all():
                row_new.at[index,'unified_grade']='C1'
            elif (row_new['unified_grade'] == 'C1').all():
                row_new.at[index,'unified_grade']='C2'
            elif (row_new['unified_grade'] == 'C2').all():
                row_new.at[index,'unified_grade']='D1'
            elif (row_new['unified_grade'] == 'D1').all():
                row_new.at[index,'unified_grade']='D2'
            elif (row_new['unified_grade'] == 'D2').all():
                row_new.at[index,'unified_grade']='E1'
            elif (row_new['unified_grade'] == 'E1').all():
                row_new.at[index,'unified_grade']='E2'
            elif (row_new['unified_grade'] == 'E2').all():
                row_new.at[index,'unified_grade']='F1'
            row_new['months_since_last_promotion']=0
            store_data = store_data.append(row_new)
            row_new.drop('global_emp_id',inplace=True,axis=1)
            pred_value1 = model.predict(row_new)
            global ab
            ab = model.predict_proba(row_new)        
            store_data.at[index,'Termination_probability'] = ab[0][1]
            store_data.at[index,'Retention_probability'] = ab[0][0]
            if pred_value1 == 0:
                retain_count = retain_count+1
            else:
                attrite_count = attrite_count+1
        text ="Employee Attrition Count before Promotion was :: {}".format(df1.shape[0])
        text1 ='Employee Attrition Count After Promotion is :: {}'.format(attrite_count)
        text2 ='Employee Retention Count After Promotion is :: {}'.format(retain_count)  
        scenario = "Scenario 2"  
        query = store_data.to_json(orient='records')
        json_data= json.loads(query)
        return jsonify(json_data)
##-------------------------------------------------------------------------------------------------------------------------------------------------------
                                                      # (2). FORM Input Format - JSON OUTPUT-FORMAT - POSTMAN
##-------------------------------------------------------------------------------------------------------------------------------------------------------
## (2c).Scenario - 3 : This scenario will update the Employee education score from 53 to 81 and make the predictions on top it - for Multiple Rows
## ## Filters on Multiple grades and sub_bu's
# You must provide three filters (POSTMAN)under Body-->Form Data for this scenario of which grade and sub_bu should be list of multiple filters 
# gender:Male,unified_grade:['A5','B1'] sub_bu:["BPS Oracle","BPS Big Data"]
##--------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/scenario3',methods=['GET','POST'])
def scenario3():
    global df1,store_data
    store_data = pd.DataFrame()
    df = data.copy()
    df1 = df.loc[(df['employee_status']==0) & (df['pred']==1)]
    if request.method=='POST':  
        gdr = request.form.get('gender')
##------------------------------------------------------------------------------------------------------------------------------------
#The below line will return single list of values from Postman whose length is '1' (output:["A5,B1,B2"] ["BPS Oracle,BPS Big Data"]     
##------------------------------------------------------------------------------------------------------------------------------------  
        grade = request.form.getlist('unified_grade')
        subbu = request.form.getlist('sub_bu')
##--------------------------------------------------------------------------------------------------------------------------------------------------------
# Formatting the grade variable by removing [ , ] , ' this will give "A5,B1,B2" and "BPS Oracle,BPS Big Data" then finally split by using comma separator 
  ## which will return output in list format like ['A5,'B1','B2'] whose length would be '3' for grade and '2' for sub_bu.
##-------------------------------------------------------------------------------------------------------------------------------------------------------   
        grade_str =str(grade)
        grade_str=grade_str.strip('[')
        grade_str=grade_str.strip(']')
        grade_str=grade_str.strip("'")
        grade_str=grade_str.split(",")
        subbu_str =str(subbu)
        subbu_str =subbu_str.strip('[')
        subbu_str =subbu_str.strip(']')
        subbu_str= subbu_str.strip("'")
        subbu_str=subbu_str.split(",")
##-------------------------------------------------------------------------------------------------------------------------------
# Inorder To Apply filters on multiple values of grade with list datatype then we must use .isna(grade_str) & .isna(subbu_str)
##-------------------------------------------------------------------------------------------------------------------------------
        df1 = df1[df1['unified_grade'].isin(grade_str)]
        df1 = df1[df1['sub_bu'].isin(subbu_str)]   
##------------------------------------------------------------------------------------------------------------------------------- 
        df1 = df1[df1['gender']==gdr]  
        df1 = df1[['global_emp_id','gender','unified_grade','city','sub_bu','ijp_90_days','tenure','bench_ageing','marital_status',
            'education_score','leaves','months_since_last_promotion','payposition_2021','tenure_in_capgemini_yr',
            'average_rating','rating_diff']]                  
        df1['education_score'] = df1['education_score'].apply(lambda x: 81 if (x == 53) else x)
        global pred_value1,attrite_count,retain_count
        # here model is nothing but Catboost  model(model) which I have created above.
        attrite_count,retain_count = 0,0
        for index,row in df1.iterrows():
            row = pd.DataFrame(row)
            row_new = row.T
            store_data = store_data.append(row_new)
            row_new.drop('global_emp_id',inplace=True,axis=1)
            pred_value1 = model.predict(row_new)
            global ab
            ab = model.predict_proba(row_new)        
            store_data.at[index,'Termination_probability'] = ab[0][1]
            store_data.at[index,'Retention_probability'] = ab[0][0]
            if pred_value1 == 0:
                retain_count = retain_count+1
            else:
                attrite_count = attrite_count+1
        text =  "Employee Attrition Count before changing Education Score was :: {}".format(df1.shape[0])
        text1 = 'Employee Attrition Count After changing Education Score is :: {}'.format(attrite_count)
        text2 = 'Employee Retention Count After changing Education Score is :: {}'.format(retain_count)    
        scenario = "Scenario 3"
        query = store_data.to_json(orient='records')
        json_data= json.loads(query)
        return jsonify(json_data)
#--------------------------------------------------------------------------------------------------------------
                         # (3). JSON Input Format - JSON OUTPUT FORMAT-POSTMAN - MULTIPLE ROWS
#-----------------------------------------------------------------------------------------------------------------------------------
##(3a).Scenario-1 :This Scenartio will Updates the Payposition If Payposition is greater than Threshold value then it will add Payposition 
                #  with Above Threshold else it will Add Payposition with below Threshold
                 # You must provide six filters in json format (POSTMAN)under Body-->raw for this scenario of which grade and sub_bu should be list of multiple filters
      # [{"gender": "Female","unified_grade": ["B1","A5"],"sub_bu":["BPS Oracle","BPS Big Data"],"Threshold":0.89,Above Threshold:0.2,Below Threshold:0.5}]
##-------------------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/scenario1_api',methods=['GET','POST'])
def scenario1_api():
    global df1,store_data,grade,subbu		
    store_data = pd.DataFrame()
    df = data.copy()
    df1 = df.loc[(df['employee_status']==0) & (df['pred']==1)]
    if request.method=='POST':  
        dat = request.json
##------------------------------------------------------------------------------------------------------------------------------------
#The below line will return single list of values from Postman whose length is '1' (output:["A5,B1,B2"] ["BPS Oracle,BPS Big Data"]     
##------------------------------------------------------------------------------------------------------------------------------------  
        grade = dat[0]['unified_grade']
        subbu = dat[0]['sub_bu']
        gdr = dat[0]['gender']
        thrs = dat[0]['Threshold']
        AT = dat[0]['Above Threshold']
        BT = dat[0]['Below Threshold'] 
##-------------------------------------------------------------------------------------------------------------------------------
# Inorder To Apply filters on multiple values of grade with list datatype then we must use .isna(grade_str) & .isna(subbu_str)
##-------------------------------------------------------------------------------------------------------------------------------
        df1 = df1[df1['unified_grade'].isin(grade)]
        df1 = df1[df1['sub_bu'].isin(subbu)]   
##---------------------------------------------------------------------------------------------------------------------------------             
        df1 = df1[df1['gender']==gdr]  
        df1 = df1[['global_emp_id','gender','unified_grade','city','sub_bu','ijp_90_days','tenure','bench_ageing','marital_status',
            'education_score','leaves','months_since_last_promotion','payposition_2021','tenure_in_capgemini_yr',
            'average_rating','rating_diff']]    
        df1['payposition_2021'] = df1['payposition_2021'].apply(lambda x: x+float(AT) if x >= float(thrs) else x+float(BT))            
        global pred_value1,attrite_count,retain_count
        # here model is nothing but Catboost  model(model) which I have created above.
        attrite_count,retain_count = 0,0
        for index,row in df1.iterrows():
            row = pd.DataFrame(row)
            row_new = row.T
            store_data = store_data.append(row_new)
            row_new.drop('global_emp_id',inplace=True,axis=1)
            pred_value1 = model.predict(row_new)
            global ab
            ab = model.predict_proba(row_new)        
            store_data.at[index,'Termination_probability'] = ab[0][1]
            store_data.at[index,'Retention_probability'] = ab[0][0]
            if pred_value1 == 0:
                retain_count = retain_count+1
            else:
                attrite_count = attrite_count+1
        text =  "Employee Attrition Count before changing Payposition was :: {}".format(df1.shape[0])
        text1 = 'Employee Attrition Count After changing Payposition is :: {}'.format(attrite_count)
        text2 = 'Employee Retention Count After changing Payposition is :: {}'.format(retain_count)    
        scenario = "Scenario 1"
        query = store_data.to_json(orient='records')
        json_data= json.loads(query)
        return jsonify(json_data)
#-----------------------------------------------------------------------------------------------------------------------------------------
                         # (3). JSON Input Format - JSON OUTPUT FORMAT-POSTMAN - MULTIPLE ROWS 
#------------------------------------------------------------------------------------------------------------------------------------------
## (3b)Scenario - 2 : This Scenario will Promotes the employee to next grade and updates months_since_last_promotion to "0" - for Multiple Rows
## Filters on Multiple grades and sub_bu's
# You must provide three filters in json format (POSTMAN)under Body-->raw for this scenario of which grade and sub_bu should be list of multiple filters 
# [{"gender": "Female","unified_grade": ["B1","A5"],"sub_bu":["BPS Oracle","BPS Big Data"]}]
##------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/scenario2_api',methods=['GET','POST'])
def scenario2_api():
    global df1,store_data
    store_data = pd.DataFrame()
    df = data.copy()
    df1 = df.loc[(df['employee_status']==0) & (df['pred']==1)]
    if request.method=='POST':  
        dat = request.json
        gdr = dat[0]['gender']
##------------------------------------------------------------------------------------------------------------------------------------
#The below line will return single list of values from Postman whose length is '1' (output:["A5,B1,B2"] ["BPS Oracle,BPS Big Data"]     
##------------------------------------------------------------------------------------------------------------------------------------  
        grade = dat[0]['unified_grade']
        subbu = dat[0]['sub_bu']
##-------------------------------------------------------------------------------------------------------------------------------
# Inorder To Apply filters on multiple values of grade with list datatype then we must use .isna(grade_str) & .isna(subbu_str)
##-------------------------------------------------------------------------------------------------------------------------------
        df1 = df1[df1['unified_grade'].isin(grade)]
        df1 = df1[df1['sub_bu'].isin(subbu)]                    
##---------------------------------------------------------------------------------------------------------------------------------  
        df1 = df1[df1['gender']==gdr]            
        df1 = df1[['global_emp_id','gender','unified_grade','city','sub_bu','ijp_90_days','tenure','bench_ageing','marital_status',
            'education_score','leaves','months_since_last_promotion','payposition_2021','tenure_in_capgemini_yr',
            'average_rating','rating_diff']]                 
        global pred_value1,attrite_count,retain_count
        # here model is nothing but Catboost  model(model) which I have created above.
        attrite_count,retain_count = 0,0
        for index,row in df1.iterrows():
            row = pd.DataFrame(row)
            row_new = row.T
            if (row_new['unified_grade'] == 'A3').all():
                row_new.at[index,'unified_grade']='A4'
            elif (row_new['unified_grade'] == 'A4').all():
                row_new.at[index,'unified_grade']='A5'  
            elif (row_new['unified_grade'] == 'A5').all():
                row_new.at[index,'unified_grade']='B1'
            elif (row_new['unified_grade'] == 'B1').all():
                row_new.at[index,'unified_grade']='B2'
            elif (row_new['unified_grade'] == 'B2').all():
                row_new.at[index,'unified_grade']='C1'
            elif (row_new['unified_grade'] == 'C1').all():
                row_new.at[index,'unified_grade']='C2'
            elif (row_new['unified_grade'] == 'C2').all():
                row_new.at[index,'unified_grade']='D1'
            elif (row_new['unified_grade'] == 'D1').all():
                row_new.at[index,'unified_grade']='D2'
            elif (row_new['unified_grade'] == 'D2').all():
                row_new.at[index,'unified_grade']='E1'
            elif (row_new['unified_grade'] == 'E1').all():
                row_new.at[index,'unified_grade']='E2'
            elif (row_new['unified_grade'] == 'E2').all():
                row_new.at[index,'unified_grade']='F1'
            row_new['months_since_last_promotion']=0
            store_data = store_data.append(row_new)
            row_new.drop('global_emp_id',inplace=True,axis=1)
            pred_value1 = model.predict(row_new)
            global ab
            ab = model.predict_proba(row_new)        
            store_data.at[index,'Termination_probability'] = ab[0][1]
            store_data.at[index,'Retention_probability'] = ab[0][0]
            if pred_value1 == 0:
                retain_count = retain_count+1
            else:
                attrite_count = attrite_count+1
        text ="Employee Attrition Count before Promotion was :: {}".format(df1.shape[0])
        text1 ='Employee Attrition Count After Promotion is :: {}'.format(attrite_count)
        text2 ='Employee Retention Count After Promotion is :: {}'.format(retain_count)  
        scenario = "Scenario 2"  
        query = store_data.to_json(orient='records')
        json_data= json.loads(query)
        return jsonify(json_data)
#-----------------------------------------------------------------------------------------------------------------------------------------
                         # (3). JSON Input Format - JSON OUTPUT FORMAT-POSTMAN - MULTIPLE ROWS 
#------------------------------------------------------------------------------------------------------------------------------------------
## (3c). Scenario - 3 : This scenario will update the Employee education score from 53 to 81 and make the predictions on top it - for Multiple Rows
## ## Filters on Multiple grades and sub_bu's
# You must provide three filters in json format (POSTMAN)under Body-->raw for this scenario of which grade and sub_bu should be list of multiple filters 
# [{"gender": "Female","unified_grade": ["B1","A5"],"sub_bu":["BPS Oracle","BPS Big Data"]}]
##--------------------------------------------------------------------------------------------------------------------------------------------
@app.route('/scenario3_api',methods=['GET','POST'])
def scenario3_api():
    global df1,store_data
    store_data = pd.DataFrame()
    df = data.copy()
    df1 = df.loc[(df['employee_status']==0) & (df['pred']==1)]
    if request.method=='POST': 
        dat = request.json 
        gdr = dat[0]['gender']
##--------------------------------------------------------------
#The below line will return single list of values from Postman 
##------------------------------------------------------------------------------------------------------------------------------------  
        grade = dat[0]['unified_grade']
        subbu = dat[0]['sub_bu']
##-------------------------------------------------------------------------------------------------------------------------------
# Inorder To Apply filters on multiple values of grade with list datatype then we must use .isna(grade_str) & .isna(subbu_str)
##-------------------------------------------------------------------------------------------------------------------------------
        df1 = df1[df1['unified_grade'].isin(grade)]
        df1 = df1[df1['sub_bu'].isin(subbu)]   
##------------------------------------------------------------------------------------------------------------------------------- 
        df1 = df1[df1['gender']==gdr]  
        df1 = df1[['global_emp_id','gender','unified_grade','city','sub_bu','ijp_90_days','tenure','bench_ageing','marital_status',
            'education_score','leaves','months_since_last_promotion','payposition_2021','tenure_in_capgemini_yr',
            'average_rating','rating_diff']]                  
        df1['education_score'] = df1['education_score'].apply(lambda x: 81 if (x == 53) else x)
        global pred_value1,attrite_count,retain_count
        # here model is nothing but Catboost  model(model) which I have created above.
        attrite_count,retain_count = 0,0
        for index,row in df1.iterrows():
            row = pd.DataFrame(row)
            row_new = row.T
            store_data = store_data.append(row_new)
            row_new.drop('global_emp_id',inplace=True,axis=1)
            pred_value1 = model.predict(row_new)
            global ab
            ab = model.predict_proba(row_new)        
            store_data.at[index,'Termination_probability'] = ab[0][1]
            store_data.at[index,'Retention_probability'] = ab[0][0]
            if pred_value1 == 0:
                retain_count = retain_count+1
            else:
                attrite_count = attrite_count+1
        text =  "Employee Attrition Count before changing Education Score was :: {}".format(df1.shape[0])
        text1 = 'Employee Attrition Count After changing Education Score is :: {}'.format(attrite_count)
        text2 = 'Employee Retention Count After changing Education Score is :: {}'.format(retain_count)    
        scenario = "Scenario 3"
        query = store_data.to_json(orient='records')
        json_data= json.loads(query)
        return jsonify(json_data)
                                ##----------------------------------------------------------------------------------------------
                                                     # (4). JSON Input Format - JSON OUTPUT-POSTMAN - MULTIPLE ROWS
                                ##----------------------------------------------------------------------------------------------
## API to Get the Predictions from Catboost Model based on JSON Input Data for Multiple rows
#  OUTPUT will be predicted with probabilities and status for No.of rows provided
##----------------------------------------------------------------------------------------------
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.json
    query = pd.DataFrame(data)  
    prediction = model.predict(np.array(query).tolist()).tolist()
    prob = model.predict_proba(np.array(query).tolist()).tolist()
    attr_prob,Status = [],[]
    for count, ele in enumerate(prob, 2):
        term_output='{0:.{1}f}'.format(ele[1], 2) 
        attr_prob.append(term_output)                   
        pred = prediction[0]
        if term_output > str(0.5):
            Status.append("Yes")      
        else:
            Status.append("No")
    return jsonify("Attrition Probability : {}".format(str(attr_prob)),"Employee Attrite : {}".format(str(Status)))
#---------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)