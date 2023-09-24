import tenseal as ts
from flask import Flask, request, jsonify, send_file
import utils
import sqlite3
import os
import collections
import glob

app = Flask(__name__)

conn = sqlite3.connect('server.db',check_same_thread=False)
c = conn.cursor()

def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS servertable(name TEXT,id INTEGER PRIMARY KEY)')

def add_data(name, id):
    c.execute('INSERT INTO servertable(name, id) VALUES (?,?)', (name,id))
    conn.commit()

def view():
    c.execute('SELECT * FROM servertable')
    data = c.fetchall()
    return data

@app.route('/')
def home():
    return "<p>Hey there!</p>"

@app.route('/register',methods = ['POST'])
def registerClient():
    create_table()
    received_json = request.get_json()
    print("Received JSON", received_json)
    add_data(received_json["name"],int(received_json["id"]))
    return jsonify({'message': 'Files uploaded successfully'})

@app.route('/getClients',methods = ['GET'])
def getClients():
    data = view()
    json_list = []
    for name, id in data:
        di = collections.defaultdict()
        di["name"] = name
        di["id"] = id
        json_list.append(di)
    print(json_list)
    return jsonify(json_list),200

@app.route('/federatedLearning')
def performFederatedLearning():
    basedir = os.path.abspath(os.path.dirname(__file__))
    n_clients = len(view())
    temp = []
    context = ts.context_from(utils.read_data(f"{basedir}/key/public.txt"))
    for i in os.listdir(f"{basedir}/uploads"):
        encrypted_proto = utils.read_data(os.path.join(basedir,"uploads", i))
        encrypted_value = ts.lazy_ckks_vector_from(encrypted_proto)
        encrypted_value.link_context(context)
        temp.append(encrypted_value)
    final_value = sum(temp) * (1/n_clients)
    basedir = os.path.abspath(os.path.dirname(__file__))
    utils.write_data(f"{basedir}/final/final_params.txt",final_value.serialize())
    return send_file(f"{basedir}/final/final_params.txt")

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PUBLIC_KEY'] = 'key'

@app.route('/uploadPublicKey', methods = ['POST'])
def upload_public_key():
    basedir = os.path.abspath(os.path.dirname(__file__))
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filename = os.path.join(basedir,app.config['PUBLIC_KEY'], file.filename)
        file.save(filename)
        return "File uploaded successfully"

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'files' not in request.files:
        return jsonify({'message': 'No files part'})
    files = request.files.getlist('files')
    if not files:
        return jsonify({'message': 'No selected files'})
    uploaded_files = []
    for file in files:
        if file.filename == '':
            continue
        basedir = os.path.abspath(os.path.dirname(__file__))
        filename = os.path.join(basedir,app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        uploaded_files.append(file.filename)
    if uploaded_files:
        return jsonify({'message': 'Files uploaded successfully'})
    else:
        return jsonify({'message': 'No files uploaded'})

if __name__ == '__main__':
    app.run(debug=True)

