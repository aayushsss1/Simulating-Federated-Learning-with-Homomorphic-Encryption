# Simulating Federated Learning with Homomorphic Encryption

This GitHub repository contains the code and resources for simulating Federated Learning with Homomorphic Encryption using a Streamlit app and Flask. This project demonstrates how to train machine learning models across clients while preserving data privacy through homomorphic encryption. The Streamlit app serves as the user interface for managing and monitoring the federated learning process

## Homomorphic Encryption
In this project, we leverage the power of [TenSEAL](https://github.com/OpenMined/TenSEAL), a cutting-edge library for homomorphic encryption. TenSEAL enables us to perform secure and privacy-preserving computations on encrypted data.

Run the below command to generate a private - public key pair in the `keys` folder.

```bash 
python3 generatekeys.py 
```

## Federated Learning

### Overall Architecture

![Alt text](image.png) {
    width: 150px;
    height: 100px;
}