# -*- coding: utf-8 -*-

from flask import Flask , render_template, request, redirect
from flask_bootstrap  import Bootstrap

from textblob import TextBlob, Word
import random
import time
import speech_recognition as sr
import nltk


from wtforms import TextField,TextAreaField, SubmitField, Form, validators

from flask_wtf import FlaskForm
import pandas as pd


from flask_cors import cross_origin
from myTextToSpeech import text_to_speech


from gtts import gTTS



import sqlite3
import numpy as np
from sklearn.externals import joblib




loaded_model=joblib.load("./pkl_objects/model.pkl")
loaded_stop=joblib.load("./pkl_objects/stopwords.pkl")
loaded_vec=joblib.load("./pkl_objects/vectorizer.pkl")




nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


app = Flask(__name__)


Bootstrap(app)
app.secret_key = 'development key'


@app.route("/", methods=["GET", "POST"])
def mainpage():
    
    return render_template('mainpage.html')

@app.route("/NLPPage", methods=["GET", "POST"])
def NLPPage():
    
    return render_template('NLPPage.html')
    
@app.route("/AudioFile", methods=["GET", "POST"])
def AudioFile():
    start_time = time.time()
    Transcription = ""
    if request.method == "POST":
        print("Facts Collected")
        
        if "record" not in request.files:
            print("record not found")
            return redirect(request.url)
        
    
        record = request.files["record"]
        
    
        if record.filename == "":
                
            print("Record Empty")
            
        if record:
            audio_recognizer = sr.Recognizer()
            audioRecord = sr.AudioFile(record)
            with audioRecord as origin:
                facts = audio_recognizer.record(origin)
            Transcription = audio_recognizer.recognize_google(facts, key=None)
            
    end_time = time.time()
    final_time = end_time-start_time
        
    if 'ConvertMe' in request.form:
        Transcription
        
        
    if 'ClearMe' in request.form:
        Transcription = "upload a new file"
                      
                
    if 'SaveMe' in request.form:
        with open("Save_Inputs.txt", 'a') as out:
            out.write(str(Transcription) + '\n')
            
        
    return render_template('NLPPage.html', Transcription=Transcription, final_time=final_time)

@app.route("/analyse", methods=["GET", "POST"])
def analyse():
    begin_time = time.time()
    if request.method == "POST":
        plaintext = request.form['plaintext']
        
        
        nlpblob = TextBlob(plaintext)
        nlpblob1 = nlpblob
        sentiment_blob,subjectivity_blob = nlpblob.sentiment.polarity, nlpblob.sentiment.subjectivity
        tokens = len(list(nlpblob.words))
        terms = list()
        for word, tag in nlpblob.tags:
            if tag == "NN":
                     terms.append(word.lemmatize())
                     words_length = len(terms)
                     words_random = random.sample(terms,words_length)
                     word_final = list()
                     for item in words_random:
                         phrase = Word(item).pluralize()
                         word_final.append(phrase)
                         phrase_summary = word_final
                         end_time = time.time()
                         time_final = end_time-begin_time
                         
                         if 'FrenchSubmit' in request.form:
                             detect = nlpblob.detect_language()
                             lang = nlpblob.translate(to='fr')
                             
                         elif 'TurkishSubmit' in request.form:
                             detect = nlpblob.detect_language()
                             lang = nlpblob.translate(to='tr')
                            
                         elif 'GermanSubmit' in request.form:
                             detect = nlpblob.detect_language()
                             lang = nlpblob.translate(to='de')
                            
                         else:
                             lang = "not translated"
                             detect = "not detected"
                            
                    
    return render_template('NLPPage.html', text = nlpblob1, tokens=tokens, sentiment_blob=sentiment_blob, subjectivity_blob=subjectivity_blob, phrase_summary=phrase_summary, time_final=time_final, lang=lang, detect=detect)


@app.route('/myText_Speech', methods=['POST', 'GET'])
@cross_origin()
def myText_Speech():
    if request.method == 'POST':
        
        text = request.form['speech']
        gender = request.form['voices']
        
        language = 'en'
        myobj = gTTS(text=text, lang=language, slow=False)
        
        
        
        if 'Speech' in request.form:
            
            text_to_speech(text, gender)
            
        if 'saveSpeech' in request.form:
            myobj.save('speech.wav')
            
        return render_template('NLPPage.html')
        
    
    
    
    return render_template('NLPPage.html')
    

    

class ContactForm(FlaskForm):
    name = TextField("Name")
    email = TextField("Email")
    subject = TextField("Subject")
    message = TextAreaField("Message")
    submit = SubmitField("Send")

@app.route('/ContactPage', methods=["GET","POST"])
def User_contact():
    UserContactForm = ContactForm()
    # here, if the request type is a POST we get the data on contat
    #forms and save them else we return the contact forms html page
    if request.method == 'POST':
        User_Name =  request.form["name"]
        User_Email = request.form["email"]
        Topic = request.form["subject"]
        Text = request.form["message"]
        res = pd.DataFrame({'name':User_Name, 'email':User_Email, 'subject':Topic ,'message':Text}, index=[0])
        res.to_csv('./contactusMessage.csv')
        print("The data is saved !")
        
        
    if 'Home' in request.form:
        return render_template('mainpage.html')
        
        
    return render_template('ContactPage.html', UserContactForm=UserContactForm)
    
@app.route('/HelpPage', methods=["GET","POST"])
def get_help():
    
    return render_template('HelpPage.html') 


def classify(document):
 label = {0: 'negative', 1: 'positive'}
 X = loaded_vec.transform([document])
 y = loaded_model.predict(X)[0]
 proba = np.max(loaded_model.predict_proba(X))
 return label[y], proba
class ReviewForm(Form):
 UserReview = TextAreaField('',[validators.DataRequired(),validators.length(min=15)])
 
def sqlite_entry(document, y):
	conn = sqlite3.connect('reviewDB.sqlite')
	c = conn.cursor()
	c.execute("INSERT INTO myreview_db (the_review, sentiment_score, the_date)"\
			" VALUES (?, ?, DATETIME('now'))", (document, y))
	conn.commit()
	conn.close()
 
@app.route('/ReviewPage')
def index():
 form = ReviewForm(request.form)
 return render_template('ReviewPage.html', form=form)

@app.route('/results', methods=['POST'])
def results():
 form = ReviewForm(request.form)
 if request.method == 'POST' and form.validate():
     UserReview = request.form['UserReview']
     
 y, proba = classify(UserReview)
 sqlite_entry(UserReview, y)
     
     
 return render_template('results.html',content=UserReview,prediction=y,probability=round(proba*100, 2))
 return render_template('ReviewPage.html', form=form)


def update_graph():

    dataSQL = [] #set an empty list
    

    conn = sqlite3.connect('reviewDB.sqlite')
    cursor = conn.cursor()
    cursor.execute("SELECT sentiment_Score,the_date FROM myreview_db")
    conn.commit()
    rows = cursor.fetchall()
    
    for row in rows:
        dataSQL.append(list(row))
        labels = ['the_review','sentiment_score','the_date']
        pd.DataFrame.from_records(dataSQL, columns=labels)
        
        
    


if __name__ == "__main__":
    app.run(debug=False)