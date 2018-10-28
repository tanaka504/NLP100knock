from flask import Flask, render_template, request, redirect, url_for
from utils import retrieve
import json, re

app = Flask(__name__)

@app.route('/')
def index():
    title = 'NLP100knock Aritist Search App'
    message = 'キーワードを入力してください'
    return render_template('index.html', title=title, message=message)

@app.route('/post', methods=['GET', 'POST'])
def post():
    title = 'Artist Search Result'
    if request.method == 'POST':
        kwds = request.form['keyword']
        try:
            type = request.form['search_type']
        except:
            message = '検索手法を選択してください'
            return render_template('index.html', title=title, message=message)

        result = retrieve(kwds, type)
        return render_template('index.html', title=title, keyword=kwds, result=result)
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
