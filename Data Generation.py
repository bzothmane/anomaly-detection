
import pandas as pd 
import random
import datetime
import string
import numpy as np
from datetime import timedelta 
from datetime import datetime, date , time
import random, string


df=pd.read_csv('/Users/home/Desktop/3DS/data_out_head_1000000.csv', sep=',',low_memory=False)
#print(df.head())
dfcities=pd.read_csv('/Users/home/Desktop/3DS/worldcities.csv', sep=',',low_memory=False)

def generateprofile(df,Mean,StDev):
    df3={}
    #event_time
    event_time = random.choice(list(df.event_time))
    df3['event_time']=event_time

    #event type
    event_type = random.choice(list(df.event_type))
    df3['event_type']=event_type

    #product_id
    iD = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    df3['product_id']=iD

    #category_id
    category_id = random.choice(list(df.category_id))
    df3['category_id']=category_id

    #category_code
    category_code = random.choice(list(df.category_code))
    df3['category_code']=category_code

    #brand
    brand = random.choice(list(df.brand))
    df3['brand']=brand

    #price
    price = random.choice(list(df.price))
    df3['price']=price

    #user_id
    usid = ''.join(random.choices(string.ascii_letters + string.digits, k=15))
    df3['user_id']=usid

    # Session_id
    ses = ''.join(random.choices(string.ascii_letters + string.digits, k=25))
    df3['Session_id']=ses

    # Customer_id
    cus= ''.join(random.choices(string.ascii_letters + string.digits, k=18))
    df3['Customer_id']=cus

    #Location
    indx=random.randint(1,41000)
    loca=[dfcities.lat[indx],dfcities.lng[indx]]
    df3['Location']=loca

    #License_id
    lic= ''.join(random.choices(string.ascii_letters + string.digits, k=25))
    df3['License_id']=lic

    ###Time

    # Log Normal Law
    N=4
    # Normal Law
    A=np.random.normal(Mean,StDev,N)
    if A[3] >= 0:
        duration=int(A[3])
    else:
        duration=1
    value = timedelta(seconds = duration*60)
    years = random.randint(2015,2021)
    months= random.randint(1,12)
    days = random.randint(1,28)
    hours= random.randint(0,23)
    minutes= random.randint(0,59)
    seconds= random.randint(0,59)
    d = date(years, months, days)
    t = time(hours, minutes,seconds)
    starttime= datetime.combine(d, t)

    #Session_start_datetime
    df3['Session_start_datetime']=starttime
    #Session_end_datetime
    endtime=starttime+value
    df3['Session_end_datetime']=endtime
    df3['duration']=duration

    #session endstar
    lenght= random.randint(10,40)
    per=timedelta(seconds = lenght*24*60*60)
    starttime= starttime-per
    durac=lenght=random.randint(2,6)
    delay = timedelta(seconds = durac*30*24*60*60)
    endtime=starttime+delay
    df3['License_start_date']=starttime
    df3['License_end_date']=endtime
    return df3
def generateprofile1(df,Mean,StDev):
    df3={}
    #event_time
    event_time = random.choice(list(df.event_time))
    df3['event_time']=event_time

    #event type
    event_type = random.choice(list(df.event_type))
    df3['event_type']=event_type

    #product_id
    iD = ''.join(random.choices(string.ascii_letters + string.digits, k=10))
    df3['product_id']=iD

    #category_id
    category_id = random.choice(list(df.category_id))
    df3['category_id']=category_id

    #category_code
    category_code = random.choice(list(df.category_code))
    df3['category_code']=category_code

    #brand
    brand = random.choice(list(df.brand))
    df3['brand']=brand

    #price
    price = random.choice(list(df.price))
    df3['price']=price

    #user_id
    usid = ''.join(random.choices(string.ascii_letters + string.digits, k=15))
    df3['user_id']=usid

    # Session_id
    ses = ''.join(random.choices(string.ascii_letters + string.digits, k=25))
    df3['Session_id']=ses

    # Customer_id
    cus= ''.join(random.choices(string.ascii_letters + string.digits, k=18))
    df3['Customer_id']=cus

    #Location
    indx=random.randint(1,41000)
    indx2=random.randint(1,41000)
    loca=[dfcities.lat[indx],dfcities.lng[indx2]]
    df3['Location']=loca

    #License_id
    lic= ''.join(random.choices(string.ascii_letters + string.digits, k=25))
    df3['License_id']=lic

    ###Time

    # Log Normal Law
    N=4
    # Normal Law
    A=np.random.normal(Mean,StDev,N)
    if A[3] >= 0:
        duration=int(A[3])
    else:
        duration=1
    value = timedelta(seconds = duration*60)
    years = random.randint(2015,2021)
    months= random.randint(1,12)
    days = random.randint(1,28)
    hours= random.randint(0,23)
    minutes= random.randint(0,59)
    seconds= random.randint(0,59)
    d = date(years, months, days)
    t = time(hours, minutes,seconds)
    starttime= datetime.combine(d, t)

    #Session_start_datetime
    df3['Session_start_datetime']=starttime
    #Session_end_datetime
    endtime=starttime+value
    df3['Session_end_datetime']=endtime
    df3['duration']=duration

    #session endstar
    lenght= random.randint(10,40)
    per=timedelta(seconds = lenght*24*60*60)
    starttime= starttime-per
    durac=lenght=random.randint(2,6)
    delay = timedelta(seconds = durac*30*24*60*60)
    endtime=starttime+delay
    df3['License_start_date']=starttime
    df3['License_end_date']=endtime
    return df3


p=0.00001
pp=0.5
    # p : proprtion de fake data
    # pp: type 1 vs type2 
numfake=int(df.shape[0]*p)
numfake1=int(numfake*pp)
numfake2=numfake-numfake1
for i in range(int(numfake1/2)):
    ligne=generateprofile(df,5,2)
    df=df.append(ligne,ignore_index = True)
for i in range(int(numfake1/2)):
    ligne=generateprofile(df,1200,500)
    df=df.append(ligne,ignore_index = True)
for i in range(int(numfake2)):
    ligne=generateprofile1(df,480,120)
    df=df.append(ligne,ignore_index = True)




print(df.shape)


    