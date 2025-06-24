import streamlit as st #Python Libraries to create webpages.
import numpy as np
import pickle

with open("iris_dataset.pkl",'rb') as f:
    model=pickle.load(f)
st.title("Iris Flower Prediction")
st.image("IRIS.png", use_container_width=True)

sepal_length=st.slider("Sepal Length(cm)",0.0,8.0)
sepal_width=st.slider("Sepal Width(cm)",0.0,8.0)
petal_length=st.slider("Petal Length(cm)",0.0,8.0)
petal_width=st.slider("Petal Width(cm)",0.0,8.0)

if st.button("Prediction"):
    input_data=np.array([[sepal_length,sepal_width,petal_length,petal_width]])
    prediction=model.predict(input_data)
    species=['setosa','versicolor','virginica']
    st.success(f"Predicted Iris Species : {species[prediction[0]]}")