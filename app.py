import joblib
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from flask import Flask, render_template, request

app = Flask(__name__)

final_features = joblib.load('final features')

model = joblib.load('FNC log_reg model')

ps = PorterStemmer()

def pred_clean(title,text):
    Textp = title + text
    review_p = re.sub('[^a-zA-z_0-9]',' ',Textp)
    review_p = review_p.lower()
    review_p = review_p.split()
    review_p = [ps.stem(word) for word in review_p if not word in stopwords.words('english')]
    dic = {}
    for i in final_features:
        if i in review_p:
            dic[i]=1
        else:
            dic[i]=0
    dic_values=list(dic.values())
    dic_values=np.array(dic_values)
    return dic_values

def prediction(dic_value):
    predict = model.predict([dic_value])
    return predict

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/", methods=['POST'])
def submit():
    
    title = request.form['title']
    text = request.form['text']
    
    predict_news = pred_clean(title, text)
   
    predict = prediction(predict_news)
                      
    if(predict==0): #fake
        return render_template('index.html', title_=title, text_=text, message="Fake", color="red")
    else:  #real
        return render_template('index.html', title_=title, text_=text, message="Real", color="green")

if __name__ == '__main__':
    app.run()
