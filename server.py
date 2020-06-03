from flask import Flask,render_template, flash,Response, request, redirect,url_for, send_from_directory, abort
from werkzeug.utils import secure_filename
import random
import sys
# Tornado web server
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
import tornado.ioloop 
from tornado.ioloop import IOLoop
from inference_model import LoadedMellotron
import os

import tornado.web

from flask_cors import CORS, cross_origin
import datetime
import hashlib
from stt import toText


UPLOAD_FOLDER = '/home/jwyang/dev/uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif', 'wav','mp3'}

ckpt_kss = "outdir_kc8/checkpoint_225500"
ckpt_you = "outdir_kc7/checkpoint_14000"
wglw = "../models/waveglow_256channels_v4.pt"
ml_kss = LoadedMellotron(ckpt_kss, wglw)
ml_you = LoadedMellotron(ckpt_you, wglw, 34)



# Initialize Flask.
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

CORS(app)

@app.route('/change/<path:modelname>')
def change_model(modelname):
    global ml_kss, wglw
    ml_kss = LoadedMellotron(modelname, wglw)
    return "good"

@app.route("/check")
def check():
    global ml_kss
    return ml_kss.ckpt


def get_hash_value(in_str, in_digest_bytes_size=8, in_return_type='hexdigest'):
    """해시값을 구한다 
    Parameter: in_str: 해싱할문자열, in_digest_bytes_size: Digest바이트크기, 
               in_return_type: 반환형태(digest or hexdigest or number) """
    assert 1 <= in_digest_bytes_size and in_digest_bytes_size <= 64
    blake  = hashlib.blake2b(in_str.encode('utf-8'), digest_size=in_digest_bytes_size)
    if in_return_type == 'hexdigest': return blake.hexdigest()
    elif in_return_type == 'number': return int(blake.hexdigest(), base=16)
    return blake.digest()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/uploads', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        print('Accept POST from client.')
        print('request.files : ', request.files)
        if 'file' not in request.files:
            flash('No file part')
            return "no file"

        print('speaker : ', request.headers.get('speaker'))
        if 'speaker' not in request.headers:
            flash('No speaker part')
            return "no speaker"
            
        file = request.files['file']
        print('&' * 10, type(file))
        speaker = request.headers.get('speaker')

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return "ㅜ"
        if file:

            rnd = random.randint(0,1000)
            title = f'{rnd}_{file.filename}'.split(".")[0]
            title = str(get_hash_value(title)) + ".wav"
            print(title)
            path = os.path.join(app.config['UPLOAD_FOLDER'], title)
            file.save(path)
            print("====[will stt]====")
            text = toText(path)

            print(text)
            print("====[will run]===")
            if speaker == "kss" :
                ml_kss.run(path,text,title, 0)
            else:
                ml_you.run(path,text,title, 0)

            return {"filename":title, "text": text}
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=text name=text>
      <input type=submit value=Upload>
    </form>
    '''


@app.route('/')
def show_entries():
    return "hello_world"


@app.route('/downloads/<path:filename>', methods=['GET'])
def download(filename):
    print("download!")
    print(filename)
    if os.path.isfile(f'/home/jwyang/dev/outputs/{filename}'):
        print("go")
        return send_from_directory(directory='/home/jwyang/dev/outputs', filename=filename)
    else:
        return abort(404)

if __name__ == "__main__":
    port = 16006
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(port)
    io_loop = tornado.ioloop.IOLoop.current()
    print("server on")
    io_loop.start()

# if __name__ == '__main__':
#     port = 16006
#     http_server = HTTPServer(WSGIContainer(app), ssl_options={
#         "certfile": "/home/jwyang/dev/localhost.crt",
#         "keyfile": "/home/jwyang/dev/localhost.key",
#     })
#     http_server.listen(port)
#     tornado.ioloop.IOLoop.instance().start()