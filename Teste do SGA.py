#!/usr/bin/env python
# coding: utf-8

# # Resultados gerais

# In[1]:


import sqlite3
import pandas as pd
import numpy as np


# In[2]:


from generate_tsp import GerarProblemaRadialTSP
from sga import SGAPermutacao, SGAInteiroLimitado


# In[3]:


def EscreverResultado(res,params,topologia,tipo,vetorOtimo):
    params['Topologia'] = topologia
    pop,stats,hof,params,custo = res
    resPerm = pd.Series(params)
    resPerm['Tipo'] = tipo
    minVal, gap, success, optSuccess = stats.select("min",'gap','success','optimal success')
    resPerm['Sucesso fraco'] = np.any(success)
    resPerm['Sucesso'] = np.any(optSuccess)
    resPerm['Geração final fraco'] = np.argwhere(success)[0][0] if np.any(success) else len(success)
    resPerm['Geração final'] = np.argwhere(optSuccess)[0][0] if np.any(optSuccess) else len(optSuccess)
    resPerm['MBF'] = np.mean(minVal)
    resPerm['Valor encontrado'] = hof[0].fitness.values[0]
    resPerm['Gap'] = custo(vetorOtimo)[0]/resPerm['Valor encontrado']
    resPerm['Indivíduo encontrado'] = hof[0]
    return pd.DataFrame([resPerm])
    
def Experimento(posicoes,params,nomePosicao='Padrão',vetorOtimo=None):
    res = SGAPermutacao(posicoes,params,vetorOtimo=vetorOtimo)
    resPerm = EscreverResultado(res,params,nomePosicao,'Permutação',vetorOtimo)
    res = SGAInteiroLimitado(posicoes,params,vetorOtimo=vetorOtimo)
    resInt = EscreverResultado(res,params,nomePosicao,'Inteiro',vetorOtimo)
    return pd.concat([resPerm,resInt],ignore_index=True)


# In[4]:


def CalcularSR(resultados):
    return (100*resultados.groupby(['Topologia','Tipo','P','mu'])['Sucesso'].sum()/resultados.groupby(['Topologia','Tipo','P','mu'])['Sucesso'].count()).unstack(['Topologia',"Tipo"])
CalcularMBF = lambda resultados: resultados.groupby(['Topologia','Tipo','P','mu'])['MBF'].describe().reset_index().set_index(['Topologia','Tipo','P','mu']).unstack(['Topologia','Tipo'])
CalcularAES = lambda resultados: resultados[resultados['Sucesso'] == True].groupby(['Topologia','Tipo','P','mu'])['Geração final'].describe().unstack(['Topologia','Tipo'])
def CalcularSRFraco(resultados):
    return (100*resultados.groupby(['Topologia','Tipo','P','mu'])['Sucesso fraco'].sum()/resultados.groupby(['Topologia','Tipo','P','mu'])['Sucesso fraco'].count()).unstack(['Topologia',"Tipo"])
CalcularAES = lambda resultados: resultados[resultados['Sucesso fraco'] == True].groupby(['Topologia','Tipo','P','mu'])['Geração final fraco'].describe().unstack(['Topologia','Tipo'])


# In[5]:


resultados = pd.DataFrame()


# In[7]:


from itertools import product

dbconn = sqlite3.connect("Resultados-finais.sqlite")

cidadesLista = [10,20,30]
individuosLista = [20,50,100]
repeticoes = 100
geracoes = 1000

listaExecucoes = list(product(list(range(repeticoes)),cidadesLista,individuosLista))
np.random.shuffle(listaExecucoes)

for repeticoes, cidades, individuos in listaExecucoes:
    resultados = pd.DataFrame()
    posicoesNormal = GerarProblemaRadialTSP(cidades)
    posicoesAleatorio = posicoesNormal.copy()
    
    resultadoOtimo = np.arange(posicoesNormal.shape[1])
    embaralhamento = resultadoOtimo.copy()
    
    np.random.shuffle(embaralhamento)
    posicoesAleatorio = posicoesNormal[:,embaralhamento].copy()
    resultadoOtimoAleatorio = np.zeros(posicoesAleatorio.shape[1], dtype=int)
    for i, val in enumerate(embaralhamento):
        resultadoOtimoAleatorio[val] = i

    params = {
        "N":geracoes,
        "P":posicoesNormal.shape[1],
        "mu":individuos,
        "cxpb": 0.7,
        "mupb":0.2,
        "muidxpb":0.05
    }
    resultados = pd.concat([resultados,Experimento(posicoesNormal,params,'Ordenada',vetorOtimo=resultadoOtimo)],ignore_index=True)
    resultados = pd.concat([resultados,Experimento(posicoesAleatorio,params,'Aleatória',vetorOtimo=resultadoOtimoAleatorio)],ignore_index=True)
    resultados.to_sql("resultados",dbconn,if_exists='append')


# In[7]:


CalcularSRFraco(resultados)


# In[8]:


resultados[(resultados['Gap']>0.9) & (resultados['Sucesso']==False)]


# In[9]:


CalcularMBF(resultados)


# In[10]:


CalcularAES(resultados)


# In[ ]:


gen, avg, min, success, gap, pop = logbook.select('gen','mean','min','success','gap','pop')

fig, ax = plt.subplots()
ax.plot(gen,avg,label="Média")
ax.plot(gen,min,label="Menor")
ax.plot(gen,success,label="Sucesso")
ax.legend()
ax.title("Estatísticas através das gerações")
plt.show()

print(np.array(pop).shape)
for i in range(np.array(pop).shape[1]):
    plt.plot(gen,np.array(pop)[:,i,0])
plt.title("Aptidões individuais")
plt.show()

