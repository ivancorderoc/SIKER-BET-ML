from flask import Flask, render_template, request
import numpy as np
import pickle
import os

app = Flask(__name__)
root_path = os.path.join(os.path.dirname(__file__), "model")
# Funcionn para cargar el modelo y el encoder desde los archivos pickle
def load_model_and_encoder():
    model_path = os.path.join(root_path, "model.pkl")
    encoder_path = os.path.join(root_path, "label_model.pkl")
    
    # Carga el modelo y el encoder usando pickle
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    with open(encoder_path, 'rb') as encoder_file:
        encoder = pickle.load(encoder_file)
    
    return model, encoder
@app.route("/")
def home():
    return render_template("homepage.html")

@app.route("/result", methods=['POST'])
def predict():
    float_features = [(x) for x in request.form.values()]

    # Cargar el modelo y el encoder
    model, encoder = load_model_and_encoder()

    # Reconvertir las tres columnas label encoded a sus representaciones originales en letras
    for i in range(-3, 0):
        encoded_value = float_features[i]
        if encoded_value in encoder.classes_:
            encoded_value = encoder.transform([encoded_value])[0]
        else:
            # Si el valor no esta; presente en el encoder, podemos manejarlo aqui;
            # Por ejemplo, podemos asignar un valor por defecto o simplemente ignorarlo
            encoded_value = 0 # Asignar un valor por defecto de ser necesario

        float_features[i] = encoded_value

    # Convertir los datos de entrada en un arreglo numpy
    final_features = np.array(float_features, dtype=float)

    # Realizar la prediccion utilizando el modelo cargado
    prediction = model.predict(final_features.reshape(1, -1))

    if prediction == 1:
        return render_template("result-1.html", prediction="THE HOME TEAM WILL WIN")
    else:
        return render_template("result-2.html", prediction="THE AWAY TEAM WILL WIN")

@app.route("/promo")
def promo():
    return render_template("promo.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/error")
def error():
    return render_template("error.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)