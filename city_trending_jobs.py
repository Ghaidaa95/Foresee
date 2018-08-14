# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 15:02:12 2018

@author: Sara
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dash
from dash.dependencies import Output, Event
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
from collections import deque
from flask import Flask
from flask import render_template
from flask import request

class city_trending_jobs:
    
    
    

  
    
    
    def __init__(self):
      self.specialty=list()  
      self.rate=list()
      self.dataframe=[]
      
      
      
    def count (self,filepath,city):
        self.dataFrame=pd.read_csv(filepath)
        newdata=self.dataFrame[self.dataFrame.job_location == city]
        newdata = newdata[newdata.job_date == 2018]
        return newdata
        
          
    def encode(self,data,column):  
       col= dict()
       for q in range(len(data)):
           x = data.iloc[q][column]
           if not data.iloc[q][column] in col:
               col.update({x:1})
           else:
                col[x]=col[x]+1    
       return col
    
    def sort (self,count):
        sorted_by_value = sorted(count, key=lambda kv: kv[1],reverse=True)
        c = 0
        for key, value in sorted_by_value:   
            if c < 10:
                self.specialty.append(key)
                self.rate.append(value)
                c = c+1
            
#    #https://medium.com/python-pandemonium/data-visualization-in-python-bar-graph-in-matplotlib-f1738602e9c4
#    def plot_bar_x(self):
#        # this is for plotting purpose
#        index = np.arange(len(self.specialty))
#        plt.bar(index,self.rate, color = 'cgmbyr')
#        plt.xlabel('Jobs', fontsize=5)
#        plt.ylabel('Growth rate', fontsize=5)
#        plt.xticks(index, self.specialty, fontsize=10, rotation=90)
#        plt.title('Trending jobs')
#        plt.savefig('graph.png', transparent=True)
#        plt.show()
#        
#        
#    plot_bar_x()   
    
    
#    app = Flask(__name__)
#     #https://pythonspot.com/flask-and-great-looking-charts-using-chart-js/
#    @app.route('/page11.html', methods = ['POST', 'GET'])
#    def chart(self):
#        #if request.method == 'POST':
#            self.specialty.clear()
#            self.rate.clear()
#            country = request.form['country']
#            new = self.count(r'C:\Users\sara\Desktop\jobDataset_All ver.2.csv', country)
#            countt=self.encode(new,'job_specialty')
#            self.sort(countt.items())
#            return render_template('page11.html', values=self.rate, labels=self.specialty)
#        
#        #return render_template('page11.html')
#     
#    if __name__ == "__main__":
#        app.run(port=4996)
    '''
    #https://www.youtube.com/watch?v=J_Cy_QjG6NE&index=1&list=PLQVvvaa0QuDfsGImWNt1eUEveHOepkjqt
    app = dash.Dash(__name__)
    app.layout = html.Div(children=[
                html.H1(children='Dash Tutorials'),
                dcc.Graph(        id='example',
                          figure={
                                  'data': [{'x': specialty, 'y': rate, 'type': 'bar', 'name': 'Cars'},],
                                  'layout': {'title': 'Trending jobs'}
                                  })
        ])
      
    
    
    if __name__ == '__main__':
        app.run_server(port = 4996)
        '''