from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

from six.moves import urllib

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

import pandas as pd

from tkinter import *

FLAGS = None

tf.logging.set_verbosity(tf.logging.INFO)

# Taxa de aprendizado do modelo
LEARNING_RATE = 0.001


def model_fn(features, targets, mode, params):

  # Conecta a primiera camada com a entrada usando Relu como função de ativação
  first_hidden_layer = tf.contrib.layers.relu(features, 10)

  # Conecta a segunda camada com a primeira camada usando Relu como função de ativação
  second_hidden_layer = tf.contrib.layers.relu(first_hidden_layer, 10)

  # Conecta a saída com a segunda camada
  output_layer = tf.contrib.layers.linear(second_hidden_layer, 1)

  # Trnsforma a saída em 1D
  predictions = tf.reshape(output_layer, [-1])
  predictions_dict = {"valores": predictions}

  # Fase de treinamente
  loss = tf.losses.mean_squared_error(targets, predictions)

  eval_metric_ops = {
      "rmse": tf.metrics.root_mean_squared_error(
          tf.cast(targets, tf.float64), predictions)
  }

  train_op = tf.contrib.layers.optimize_loss(
      loss=loss,
      global_step=tf.contrib.framework.get_global_step(),
      learning_rate=params["learning_rate"],
      optimizer="SGD")

  return model_fn_lib.ModelFnOps(
      mode=mode,
      predictions=predictions_dict,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops)


def main(acao):
  # Configurando os parametros
  model_params = {"learning_rate": LEARNING_RATE}

  # Inicializando o estimador
  nn = tf.contrib.learn.Estimator(model_fn=model_fn, params=model_params, model_dir='./modelo_rna')

  def load_weather_frame(filename):
    #carrega os dados
    data_raw = pd.read_csv(filename, dtype={'Data_OB': str})

    valor = []

    # Divide o valor por 100000000 para diminuir a dimensão
    for index, row in data_raw.iterrows():
      row['Valor Total'] = row['Valor Total'].replace('.','').replace(',','.')
      valorU = float(row['Valor Total'])
      valorU = valorU/100000000
      valor.append(valorU)

    data_raw['_valor'] = pd.Series(valor, index=data_raw.index)
    
    times_ano = []
    times_mes = []

    # Separa os meses e os anos das datas
    for index, row in data_raw.iterrows():
      data_div = row['Data_OB'].split('/')
      dia = put_zero(data_div[0])
      mes = put_zero(data_div[1])
      ano = data_div[2]
      if(len(data_div[2]) <= 2):
        ano = '20' + ano
      times_ano.append(ano)
      times_mes.append(mes)
    data_raw['_ano'] = pd.Series(times_ano, index=data_raw.index)
    data_raw['_mes'] = pd.Series(times_mes, index=data_raw.index)

    data_raw['_valor'] = data_raw['_valor'].astype(float)

    data_raw = data_raw.groupby([data_raw['_ano'],data_raw['_mes']]).sum().reset_index()

    #data_raw.to_csv('data/data.csv', index = False)

    data_raw['_mes'] = data_raw['_mes'].astype(float)
    data_raw['_ano'] = data_raw['_ano'].astype(float)

    # Cria os vetores dos valores de X (mês e ano) e Y (valor)
    df =  pd.DataFrame(data_raw, columns=['_mes', '_ano'])
    df2 = pd.DataFrame(data_raw, columns=['_valor'])

    # Retorna um vetor com unidades de mês e anos normalizados, e um outro vetor com os valores
    return (df - df.mean()) / (df.max() - df.min()), df2

  # Formata do dia ou mês
  def put_zero(dia):
  	if len(dia) == 1:
  		return'0'+dia
  	return dia

  # Lê os dados e separa em treinamento e teste
  x, y = load_weather_frame("data/BASE_SUS_2009_2016.csv")
  x_train = x.loc[:93].values
  y_train = (np.array(y.loc[:93].values)).ravel()
  x_teste = x.loc[94:].values
  y_teste = (np.array(y.loc[94:].values)).ravel()

  # Inicializa os valores das entradas para treinamento
  def get_train_inputs():
    x = tf.constant(x_train)
    y = tf.constant(y_train)
    return x, y

  # Executa o treinamento  
  if(acao == 'treinamento'):
  	nn.fit(input_fn=get_train_inputs, steps=10000)

  # Calcula o erro médio quadrado 
  def get_test_inputs():
    x = tf.constant(x_teste)
    y = tf.constant(y_teste)
    return x, y
  
  ev = nn.evaluate(input_fn=get_test_inputs, steps=1)
  print("Loss: %s" % ev["loss"])
  print("Root Mean Squared Error: %s" % ev["rmse"])

  # Mostra os resultados
  predictions = nn.predict(x=x_teste, as_iterable=True)
  a = []
  previsão_label = []
  for i, p in enumerate(predictions):
    print("Previsão %s: %s" % (i + 1, p["valores"]*100000000)) # Multiplica o resultado pro 100000000 para mostrar o valor na escala real
    print("Valor %s: %s" % (i + 1, y_teste[i]*100000000)) # Multiplica o resultado pro 100000000 para mostrar o valor na escala real
    previsão_label.append('Previsão ' + str(i+1) +':    ' + str(p["valores"]*100000000) + '\n'+ 'Valor ' + str(i+1) +':    ' + str(y_teste[i]*100000000))
    a.append(p["valores"])
  print('Erro: ', np.mean(np.abs((y_teste - a) / y_teste)) * 100, '%')

  #Incluindo os valores para a interface gráfica
  erro = 'Erro: ' + str(np.mean(np.abs((y_teste - a) / y_teste)) * 100) + '%'
  v.set(erro)
  p1.set(previsão_label[0])
  p2.set(previsão_label[1])

#Chamada de teste ou previsão
def teste():
  main('teste')

#Chamada para treinamento
def treinamento():
  main('treinamento')

#Criação da interface gráfica
root = Tk()
root.title("Rede Neural - Previsão do SUS")

#Inserindo os botoões
button_1 = Button(root, text = 'Teste', command= teste)
button_1.pack()

button_2 = Button(root, text = 'Treinamento', command = treinamento)
button_2.pack()

#Inserindo os labels de previsão e erro
p1 = StringVar()
p1.set("Previsão 1: ")
label_p1 = Label(root, textvariable=p1)
label_p1.pack()

p2 = StringVar()
p2.set("Previsão 2: ")
label_p2 = Label(root, textvariable=p2)
label_p2.pack()

v = StringVar()
v.set("Erro: ")
w = Label(root, textvariable=v)
w.pack()

root.mainloop()