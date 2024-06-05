import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sga import SGAInteiroLimitado, SGAPermutacao
from generate_tsp import GerarProblemaRadialTSP

import sqlite3


def TestarTaxaDeMutacao():
    """
    Testar taxa de sucesso entre taxas altas e baixas de mutação
    """
    dbconn = sqlite3.connect("TaxasMutacao.sqlite")
    posicoes = GerarProblemaRadialTSP(21)
    
    valorOtimo = np.sum(np.linalg.norm(posicoes - np.roll(posicoes,1,axis=1),axis=0))

    params = {
            "N": 1000,
            "P": posicoes.shape[1],
            "mu": 100,
            "cxpb": 0.3,
            "mupb": 0.3,
            "muidxpb": 0.3
    }

    idx = 0
    for cxpb,mupb in [(0.3,0.3),(0.3,0.7),(0.7,0.3),(0.7,0.7)]:
        params['cxpb'] = cxpb
        params['mupb'] = mupb
        for i in range(100):

            pop, stats, hof, params, cost = SGAPermutacao(posicoes,params)
            
            min,suc,gen = stats.select('min','success','gen')
            params['Tipo'] = "Permutação"
            params['Geração final'] = len(gen)
            params['Sucesso'] = suc[-1]
            params['MBF'] = np.mean(min)
            params['Resultado'] = hof[0].fitness.values
            params['Gap'] = hof[0].fitness.values/valorOtimo
            pd.DataFrame(params).to_sql("resultados",dbconn,if_exists='append')
            
            pop, stats, hof, params, cost = SGAInteiroLimitado(posicoes,params)
            
            min,suc,gen = stats.select('min','success','gen')
            params['Tipo'] = "Inteiro"
            params['Geração final'] = len(gen)
            params['Sucesso'] = suc[-1]
            params['MBF'] = np.mean(min)
            params['Resultado'] = hof[0].fitness.values
            params['Gap'] = hof[0].fitness.values/valorOtimo
            pd.DataFrame(params).to_sql("resultados",dbconn,if_exists='append')
            print(idx) 
            idx += 1

if __name__ == "__main__":
    TestarTaxaDeMutacao()
