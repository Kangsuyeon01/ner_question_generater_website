# app.py
import sys
import os
sys.path.append('/home/suyeon/code/bert_ner')
from flask import Flask, render_template, request
from question_generater import start


#Flask 객체 인스턴스 생성
app = Flask(__name__)

@app.route('/') # 접속하는 url
def index():
  return render_template('index.html')



@app.route('/news_result', methods = ['POST', 'GET'])
def news_content_crawl():
    if request.method == 'POST':
        context =  request.form["input_context"]
        empty_questions,empty_answers, ox_questions, ox_answers = start(context)
        questions = [empty_questions,empty_answers, ox_questions, ox_answers]

    return render_template("news_result.html", questions = questions)


if __name__=="__main__":
  # app.run(debug=True)
  # host 등을 직접 지정하고 싶다면
  app.run(host="127.0.0.1", port="5555", debug=True)