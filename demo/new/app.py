# import required libraries
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
import re
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
kw_model = KeyBERT()
import operator
import pickle
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import operator
import math
nltk.download('punkt')
from nltk.corpus import stopwords
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
import spacy
from sklearn import preprocessing
#from similarity.jarowinkler import JaroWinkler
#from similarity.cosine import Cosine
wordnet_lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english')) 
from nltk.stem import PorterStemmer
from rank_bm25 import *
ps = PorterStemmer()
from sklearn.metrics.pairwise import cosine_similarity
#from similarity.jarowinkler import JaroWinkler

#from similarity.cosine import Cosine
#jarowinkler = JaroWinkler()

# initialize the flask app
app = Flask(__name__)

# create the upload folder to store input files
upload_folder = "uploads/"
if not os.path.exists(upload_folder):
    os.mkdir(upload_folder)

# configure the upload folder
app.config["UPLOAD_FOLDER"] = upload_folder

# configure allowed file types
allowed_extentions = ['csv', 'tsv', 'txt']

def check_file_extension(filename):
    return filename.split('.')[-1] in allowed_extentions

# path for uploading input files
@app.route('/')
def upload_file():
    return render_template('upload.html')


@app.route('/upload', methods = ['GET', 'POST'])
def save_file():
    if request.method == 'POST':
        files = request.files.getlist('files')
        print(files)
        for f in files:
            print(f.filename)

            #save file
            if check_file_extension(f.filename):
                f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        #f1 = request.files['file1']
        #filename = secure_filename(f1.filename)
        #f1.save(app.config['UPLOAD_FOLDER'] + filename)
        return 'file upload successfully'

        
if __name__ == '__main__':
    app.run(host="localhost", port=5000, debug = True)