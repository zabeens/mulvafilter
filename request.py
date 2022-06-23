import requests

url = 'http://127.0.0.1:6000/predict_api'
r = requests.post(url,json={'gender':6, 'unified_grade':2,'city':12,'sub_bu':20,
'ijp_90_days':4,'Tenure':4,'bench_ageing':5',marital_status':10,'education_score':4,'months_since_last_Promotion':6,
                      'Payposition_2021':6,'tenure_in_capgemini_yr':4,
                      'Leaves_takes_90 days':6,'Average_Rating':4,'rating_diff':3})
print(r.json())