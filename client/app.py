import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Variable
import numpy as np
import pandas as pd
import tenseal as ts
import utils
import requests
import collections
import os
import streamlit as st
from client import Client, LogisticRegression, scale_dataset

import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()
basedir = os.path.abspath(os.path.dirname(__file__))

def create_table():
    c.execute('CREATE TABLE IF NOT EXISTS clienttable(name TEXT,id INTEGER PRIMARY KEY,filepath TEXT)')

def add_data(name, id, filepath):
    c.execute('INSERT INTO clienttable(name, id, filepath) VALUES (?,?,?)', (name,id,filepath))
    conn.commit()

def view():
    c.execute('SELECT * FROM clienttable')
    data = c.fetchall()
    return data

def main():
    menu = ["Client","Server","Predict"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Client":
        st.title("Federated Learning using Homomorphic Encryption")
        st.subheader("Enter Client Details")
        with st.form(key='form1'):
            name = st.text_input("Name")
            id   = st.text_input("ID")
            uploaded_file = st.file_uploader("Upload a CSV File")
            if uploaded_file is not None:
                with open(os.path.join(f"{basedir}/data/",uploaded_file.name),"wb") as f:
                    f.write(uploaded_file.getbuffer()) 
                st.success("Saved File:{} to data/".format(uploaded_file.name))
            submit_button = st.form_submit_button(label='Register')

        if submit_button:
            client_list = collections.defaultdict()
            client_list["id"] = id
            client_list["name"] = name
            response = requests.post('http://127.0.0.1:5000/register',json=client_list)
            if response.status_code == 200:
                st.success("Client Registered!")
            if uploaded_file is not None:
                client_list["uploaded_file"] = basedir + "/data/" + uploaded_file.name
                with open(client_list["uploaded_file"], 'wb') as f:
                    f.write(uploaded_file.read())
            add_data(client_list["name"],client_list["id"],client_list["uploaded_file"])

        local_data = view()
        st.subheader("Registered Clients")
        for i in local_data:
            with st.expander(i[0]):
                dataframe = pd.read_csv(i[2])
                st.write(dataframe)

        with st.form(key='form2'):
            training = st.form_submit_button(label='Perform Local Training')
            files = []
            if training:
                with st.spinner('Wait for it...'):
                    load_clients = view()
                    clients = []
                    for i in load_clients:
                        clients.append(Client(i[0],i[2],f"{basedir}/outputs/enc_weight_client{i[1]}.txt",n_features=31, iters=10))
                    losses, accuracies = local_training(clients)

                    for i in load_clients:
                        files.append(('files', (f'file{i[1]}.txt', open(f'{basedir}/outputs/enc_weight_client{i[1]}.txt', 'rb'))))
                    
                    print(files)
                    response = requests.post('http://127.0.0.1:5000/upload',files=files)
                    print("Response", response)
                    print("Losses - ", losses)
                    print("Accuracies - ", accuracies)
                st.success("Done!")
    
    if choice == 'Server':
        st.title("Server")
        st.subheader("Registered Clients")
        clients_info = requests.get("http://127.0.0.1:5000/getClients")
        st.write(pd.DataFrame.from_dict(clients_info.json()))

        st.subheader("Upload Public Key")
        with st.form(key='form3'):
            uploaded_file = st.file_uploader("Upload the Public Key txt file")
            if uploaded_file:
                st.write("File uploaded successfully!")
            submit_button = st.form_submit_button(label='Send File to Server')
        
        if submit_button:
            files = {'file': (uploaded_file.name, uploaded_file.read())}
            try:
                response = requests.post('http://127.0.0.1:5000/uploadPublicKey', files=files)
                if response.status_code == 200:
                    st.success("Key sent to server successfully!")
                else:
                    st.error(f"Failed to send Key. Status code: {response.status_code}")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        
        with st.form(key='form4'):
            training = st.form_submit_button(label='Perform Federated Learning')
            files = []
            if training:
                with st.spinner('Wait for it...'):
                    response = requests.get('http://127.0.0.1:5000/federatedLearning',files=files)
                    file_content = response.content
                    with open(f"{basedir}/outputs/updated_weight.txt", 'wb') as local_file:
                        local_file.write(file_content)
                st.success("Model Updated!")
    
    if choice == 'Predict':
        st.title("Predict")
        global_model = decrypted_model()
        st.subheader('Model parameters')
        st.markdown('**Weights** \n```%s' % global_model.linear.weight) ### Virtualize record of training processlobal_model.linear.weight)
        st.markdown('**Bias** \n```%s' % global_model.linear.bias)
        st.subheader('Upload Test Data')
        with st.form(key='predict'):
            uploaded_file = st.file_uploader("Upload a CSV File")
            if uploaded_file is not None:
                file = basedir + "/data/" + uploaded_file.name
                with open(file, 'wb') as f:
                    f.write(uploaded_file.getbuffer()) 
                st.success("Saved File:{} to /data/".format(uploaded_file.name))
            predict = st.form_submit_button(label='Predict')
        
        if predict:
            df_test = pd.read_csv(file)
            df_test["diagnostic"] = (df_test["diagnostic"] == "M").astype(int)
            test , X_test , Y_test  = scale_dataset(df_test , False)
            test_acc = compute_federated_accuracy(global_model, X_test, Y_test)
            to_percent = lambda x: '{:.2f}%'.format(x)
            st.write('\nTesting Accuracy = {}'.format(to_percent(test_acc)))

def local_training(clients):

    n_clients = len(clients)
    # record losses and accuracies report from clients
    losses = [[] for i in range(n_clients)]
    accuracies = [[] for i in range(n_clients)]

    # perform local training for each clients then report acc and loss to server
    for i in range(n_clients):
        clients[i].local_training(debug=False)    
        # report to server
        losses[i].append(clients[i].losses[-1])
        accuracies[i].append(clients[i].accuracies[-1])
    
    # clients encrypt the final weights of local model after training
    for i in range(n_clients):
            clients[i].encrypted_model_params()
    
    return losses, accuracies


def decrypted_model():
    context = ts.context_from(utils.read_data("keys/secret.txt"))
    encrypted_proto = utils.read_data(f"{basedir}/outputs/updated_weight.txt")
    encrypted_value = ts.lazy_ckks_vector_from(encrypted_proto)
    encrypted_value.link_context(context)
    federated_model_params = encrypted_value.decrypt()
    # convert float to tensor context
    W = Variable(torch.tensor([federated_model_params[:-1]], dtype = torch.float32))
    B = Variable(torch.tensor( federated_model_params[-1], dtype = torch.float32))

    global_model = LogisticRegression(31)
    global_model.linear.weight = nn.Parameter(W)
    global_model.linear.bias   = nn.Parameter(B)

    return global_model

def compute_federated_accuracy(model, input, output):
    prediction = model(input)
    n_samples = prediction.shape[0]
    s = 0.
    for i in range(n_samples):
        p = 1. if prediction[i] >= 0.5 else 0.
        e = 1. if p == output[i] else 0.
        s += e
    return 100. * s / n_samples

create_table()
main()