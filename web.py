
# coding: utf-8

import pandas as pd 
from prediction1 import prediction_Class
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pylab as plt
from city_trending_jobs import city_trending_jobs
from prediction_salary import salary
from collab import collab

from flask import Flask
from flask import render_template
from flask import request





dataset=pd.read_csv("data/jobDataset_All.csv")
    
dataset


pre_cols=["job_date"]
target="count"

x = dataset.job_title.value_counts().nlargest(10).sort_index()


name=list()
count =list()
for a in x.index:
   name.append(a)
   count.append(x[a])
   
   

app = Flask(__name__)

@app.route('/index.html')
def index():   
    return render_template('index.html', values=count, labels=name)
    
 #https://pythonspot.com/flask-and-great-looking-charts-using-chart-js/
@app.route('/page1.html', methods = ['POST', 'GET'])
def chart():
    if request.method == 'POST':
        country = request.form['country']
        spe = request.form['specialty']                
        p =prediction_Class(dataset,pre_cols,target)
        dates=list()
        count=list()
        dates.clear()
        count.clear()
        p.cou=""
        p.dates=""
        p.predict(spe,country)
        print(country)        
        return render_template('page1.html',  scroll= 'con', values=p.cou, labels=p.dates, spe = spe, country = country)


    return render_template('page1.html')


@app.route('/reco.html', methods = ['POST', 'GET'])
def reco():
    if request.method == 'POST':
        
        a = float(request.form['Tableau'])
        b= float(request.form['PPC'])
        c=float( request.form['Adobe'])
        d=float( request.form['Microsoft'])
        e=float( request.form['Javascript'])
        f= float(request.form['Excel'])
        g= float(request.form['social_tools'])
        h= float(request.form['HTML'])
        i= float(request.form['Wordpress'])
        j= float(request.form['ga'])
        k=float( request.form['SQL'])
        l=float( request.form['SEO'])
        m= float(request.form['Stats'])
        n= float(request.form['CMS'])
        o= float(request.form['Email'])
        p= float(request.form['CSS'])
        q= float(request.form['Leadership'])
        r= float(request.form['Passion'])
        s=float( request.form['Teamwork'])
        t= float(request.form['Communication'])
        u= float(request.form['Writing'])
        v= float(request.form['Presentation'])
        w= float(request.form['Detail'])
        x= float(request.form['Creative'])
        y= float(request.form['Paid'])
        z= float(request.form['pr'])
        aa= float(request.form['Sales'])
        bb= float(request.form['Content'])
        cc= float(request.form['Analytics'])
        dd= float(request.form['Social_media'])
        ee= float(request.form['Digital'])
        
        #c = collab(Tableau,PPC,Adobe,Microsoft,Javascript,Excel,social_tools,
         #         HTML,Wordpress,ga,SQL,SEO,Stats,CMS,Email,CSS,Leadership,
         #         Passion,Teamwork,Communication,Writing,Presentation,Detail,
          #        Creative,Paid,pr,Sales,Content,Analytics,Social_media,Digital)
        
        c=collab(a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,bb,cc,dd,ee)
        c.pearson()       
        return render_template('reco.html',scroll='result', similar_jobTitle = c.similar_jobTitle, similar_jobDescription = c.similar_jobDescription)


    return render_template('reco.html')

@app.route('/page11.html', methods = ['POST', 'GET'])
def trend_city():
    if request.method == 'POST':
        country = request.form['country']    
        c = city_trending_jobs()
        c.specialty.clear()
        c.rate.clear()
        new = c.count(r'data\jobDataset_All ver.2.csv', country)
        countt=c.encode(new,'job_specialty')
        c.sort(countt.items())    
        return render_template('page11.html', scroll= 'sc',values = c.rate, 
                               labels = c.specialty, country = country)
    return render_template('page11.html')



@app.route('/page2.html')
def page2():    
    return render_template('page2.html')

@app.route('/page3.html',methods = ['POST', 'GET'])
def page3():    
    if request.method == 'POST':
        spe = request.form['specialty']
        p=salary()
        p.name(spe)        
        return render_template('page3.html',  scroll= 'con', data = p.data , spe = spe)
    return render_template('page3.html')

@app.route('/page4.html')
def page4():    
    return render_template('page4.html')



 
if __name__ == "__main__":
    app.run(port=4996, use_reloader=True)

