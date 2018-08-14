import math
import pandas as pd

class collab:
    
    def __init__(self,a=0,b=0,c=0,d=0,e=0,f=0,g=0,h=0,i=0,j=0,k=0,l=0,m=0,n=0,o=0,p=0,q=0,r=0,s=0,t=0,u=0,v=0,w=0,x=0,y=0,z=0,aa=0,bb=0,cc=0,dd=0,ee=0):
        self.tableau=a
        self.ppc=b
        self.adob=c
        self.microsoft=d
        self.javascript=e
        self.excel=f
        self.social_tools=g
        self.html=h
        self.wordpress=i
        self.general_availability=j
        self.sql=k
        self.seo=l
        self.stats=m
        self.cms=n
        self.email=o
        self.css=p
        self.leadership=q
        self.passion=r
        self.teamwork=s
        self.communication=t
        self.writing=u
        self.presentation=v
        self.detail=w
        self.creative=x
        self.paid=y
        self.pr=z
        self.sales=aa
        self.content=bb
        self.analytics=cc
        self.social_media=dd
        self.digital=ee
        self.similar_jobTitle=''
        self.similar_jobDescription=''
    def prepare(self,df):
        dic=list()
        for i in range(len(df)):
            li=list()
            for j in range(4,len(df.columns)):
                row=df.loc[i][j]
                li.append(float(row))
            dic.append((df.loc[i][3],li))
        return dic
    def pearson(self):
        print("")
        userSkills=list()
        userSkills.append(self.adob)
        userSkills.append(self.analytics)
        userSkills.append(self.cms)
        userSkills.append(self.communication)
        userSkills.append(self.content)
        userSkills.append(self.creative)
        userSkills.append(self.css)
        userSkills.append(self.detail)
        userSkills.append(self.digital)
        userSkills.append(self.email)
        userSkills.append(self.excel)
        userSkills.append(self.general_availability)
        userSkills.append(self.html)
        userSkills.append(self.javascript)
        userSkills.append(self.leadership)
        userSkills.append(self.microsoft)
        userSkills.append(self.paid)
        userSkills.append(self.passion)
        userSkills.append(self.ppc)
        userSkills.append(self.pr)
        userSkills.append(self.presentation)
        userSkills.append(self.sales)
        userSkills.append(self.seo)
        userSkills.append(self.social_media)
        userSkills.append(self.social_tools)
        userSkills.append(self.sql)
        userSkills.append(self.stats)
        userSkills.append(self.tableau)
        userSkills.append(self.teamwork)
        userSkills.append(self.wordpress)
        userSkills.append(self.writing)
       
        
        df=pd.read_csv(r'data\marketing-internship-postings-QueryResult.csv')
        
       
        
        jobSkillsList=self.prepare(df)
        
        jobSkills=dict()
        jobTitle=dict()
        for i in range(len(jobSkillsList)):
            jobTitle.update({i:jobSkillsList[i][0]})
            jobSkills.update({i:jobSkillsList[i][1]})
            
        jobSkills.update({len(jobSkills):userSkills})
#        
        def pearson_score(p1, p2):
            n = len (p1)
            sum1 = 0
            sum_p1 = 0
            sum_p2 = 0
            sq_sum_p1 = 0
            sq_sum_p2 = 0
            for i in range(n):
                sum1 += p1[i]*p2[i]
                sum_p1 += p1[i]
                sum_p2 += p2[i]
                sq_sum_p1 += p1[i]**2
                sq_sum_p2 += p2[i]**2
                if(sq_sum_p1==0):
                    sq_sum_p1=1
                if(sq_sum_p2==0):
                    sq_sum_p2=1
            num = (n*sum1)-(sum_p1*sum_p2)
            denom = math.sqrt((n*sq_sum_p1)-(sum_p1**2))*math.sqrt((n*sq_sum_p2)-(sum_p2**2))
            return num/denom
        
        def similar_user_pearson_score(user_rating, user):
            user1 = jobSkills[user]
            min_distanse = -1 
            similar_user = None 
            for u in user_rating:
                if u == user:
                    continue
                user2 = user_rating[u]
                dist = pearson_score(user1,user2)
                if dist > min_distanse:
                    min_distanse = dist
                    similar_user = u
            return similar_user
        
        user = len(jobSkills)-1
        similar_user_id = similar_user_pearson_score(jobSkills,user)
        self.similar_jobTitle=jobTitle[similar_user_id]
        self.similar_jobDescription=df['description'][similar_user_id]
    