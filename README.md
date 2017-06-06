# Projeto IA - Previsão do repasse do SUS - Redes Neurais

Esta é uma rede neurais criada através do Tensor Flow, que tem como objetivo fazer a previsão do repasse mensal para SUS de Pernambuco.

## Pré-requisitos

Para executar a rede neural, é necessário a instalação de algumas bilbiotecas. 

```
pip3 install -r requirements.txt
```

## Treinando e/ou Testando a rede

A rede está inicialmente treinada, mas você pode continuar o treino ou fazer o um treino novo, para isso é necessário apagar a pasta modelo_rna.
Para iniciar o trenamento, execute:

```
python3 RNA.py treinamento
```

Caso só deseje testar a rede para os dois últimos meses da base de dados, execute:

```
python3 RNA.py teste
```


## Treinando e/ou Testando a rede com a interface gráfica

Caso deseje treinar e/ou testar a rede neural através da interface gráfica, execute:

```
python3 RNA_GUI.py
```


## Built With

* [TensorFlow](https://www.tensorflow.org/programmers_guide/) - Biblioteca open source para machine learning
