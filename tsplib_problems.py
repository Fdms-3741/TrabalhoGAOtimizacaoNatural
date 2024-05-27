import re
from re import search 
from pathlib import Path 

import pandas as pd 
import numpy as np


def LerArquivo(nome):

    file = Path(nome)
    
    if file.suffix != ".tsp":
        return None
    
    data = {}
    with file.open() as fd:
        while True:
            input = fd.readline().replace('\n','')
            # Divisão para os arquivos
            termos = None
            for pattern in ["\s:\s", ":\s","\s:", ":"]:
                res = search(pattern,input)
                if res:
                    termos = input.split(pattern.replace('\s',' '))
                    break
            if termos is None:
                break
            data[termos[0].replace(' ','')] = termos[1]
    return data

def LerResultado(nomeResultado,solutionsPath='./tsplib/solutions'):
    with open(solutionsPath) as fd:
        while True:
            try:
                nomeArquivo, solucao = fd.readline().split(' : ')
            except Exception:
                break
            if nomeArquivo == nomeResultado:
                return float(solucao)

def GetProblemsDataFrame(problemPath='.'):
    results = []
    for file in Path(problemPath).iterdir():   
        if not (res := LerArquivo(file)) is None:
            results.append(res)

    # Encontrar problemas de diferentes escalas
    results = pd.DataFrame(results)
    results['DIMENSION'] = results['DIMENSION'].astype(np.int64)
    results['ORDER'] = np.floor(np.log10(results['DIMENSION'])).astype(np.int64)
    results = results.sort_values('DIMENSION',ascending=False)
    results['OPTIMAL_VALUE'] = np.nan
    return results

def GetParametersProblem(name,sourceDir='./tsplib/'):

    results = []
    # Verifica se o arquivo existe
    if (file := Path(sourceDir+name+".tsp")).exists():
        # Abre o arquivo
        with file.open() as fd:
            data = False # Indica que está lendo uma linha de dados
            idx = 0
            # Itera através das linhas 
            while True:
                try:
                    input = fd.readline().replace('\n','')
                except:
                    break
                # Detecta quando entra na seção de dados:
                if input == "NODE_COORD_SECTION":
                    data=True
                    continue

                # Detecta quando acaba o arquivo
                if input == "EOF":
                    break
                
                # Escreve os dados na matriz
                if data:
                    input = re.sub('^\s+','',input,count=1)
                    idStr, x, y = re.sub('\s+',',',input).split(',')
                    results.append(np.array([float(x),float(y)]))
                    idx += 1
                    

        results = np.stack(results)
    else:
        raise Exception(f"Arquivo {name}.tsp não existe")
    
    return results

def GerarProblemaTSPLIB(indice):
    # Le as características do problema em tsplib
    results = GetProblemsDataFrame('./tsplib/')
    # Filtra resultados pelo tipo euclidiano
    results = results[results['EDGE_WEIGHT_TYPE']=='EUC_2D']
    results = results.groupby('ORDER').apply(lambda x: x.iloc[0])
    results = results.set_index('NAME')
    # Adicionar valores ótimos ao resultado
    for i in results.index:
        valorOtimo = LerResultado(i)
        results.at[i,'OPTIMAL_VALUE'] = valorOtimo
    # Retorna pd.Series com características do problema e np.array com dimensões do problema
    return results.iloc[indice], GetParametersProblem(results.index[indice])

def main():
    for i in range(4):
        data,_ = GerarProblemaTSPLIB(i)
        print(data)

if __name__ == "__main__":
    main()
