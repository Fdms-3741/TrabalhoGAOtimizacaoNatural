import array 
import random 

import numpy as np
from deap import creator, base, tools, algorithms

# Definição da função de aptidão e dos tipos dos invidívuos
creator.create("FitnessMin",base.Fitness,weights=(-1.0,)) # Minimiza
creator.create("individual",array.array,typecode='i',fitness=creator.FitnessMin)

def CriarAlgoritmoSGAInteiroLimitado(TAMANHO_INDIVIDUO,PROBABILIDADE_MUTACAO_INDICE,FuncaoCusto):

    toolbox = base.Toolbox()
    # Atributo do indivíduo
    toolbox.register("atributo",random.randint,0,TAMANHO_INDIVIDUO-1)
    ##toolbox.register("atributo", random.sample, range(IND_SIZE), IND_SIZE)
    # Indivíduo
    toolbox.register("individual",tools.initRepeat,creator.individual,toolbox.atributo,n=TAMANHO_INDIVIDUO)
    # População 
    toolbox.register("population",tools.initRepeat,list,toolbox.individual)

    # Seleção de indivíudos
    toolbox.register("select",tools.selTournament,tournsize=3)
    # Função de recombinação 
    toolbox.register('mate',tools.cxPartialyMatched)
    # Função de mutação
    toolbox.register('mutate',tools.mutUniformInt,low=0,up=TAMANHO_INDIVIDUO-1,indpb=PROBABILIDADE_MUTACAO_INDICE)
    # Função de avaliação
    toolbox.register('evaluate',FuncaoCusto)

#    def CorrecaoInteirosFronteira(valorMaximo):
#        def decorator(func):
#            def wrapper(*args, **kargs):
#                offspring = func(*args, **kargs)
#                for child in offspring:
#                    for i in range(len(child)):
#                        child[i] = (np.rint(child[i])%valorMaximo).astype('int64') 
#                return offspring
#            return wrapper
#        return decorator

#    toolbox.decorate('mutate',CorrecaoInteirosFronteira(TAMANHO_INDIVIDUO))
    
    return toolbox 


def CriarAlgoritmoSGAPermutacao(TAMANHO_INDIVIDUO,PROBABILIDADE_MUTACAO_INDICE,FuncaoCusto):

    toolbox = base.Toolbox()
    # Cria uma permutação aleatória entre 0 e N-1
    toolbox.register("indices",random.sample, range(TAMANHO_INDIVIDUO),TAMANHO_INDIVIDUO)
    # Registro de geração de um indivíduo na toolbox 
    toolbox.register(
        "individual", # Define o indivíduo
        tools.initIterate, # Função que pega uma função geradora e coloca os resultados em um tipo 
        creator.individual, # Tipo que recebe os resultados
        toolbox.indices # Função geradora de dados para o indivíduo
    )

    # Registra a construção de populações
    toolbox.register(
        "population",
        tools.initRepeat, # Função que inicializa uma população repetindo chamadas a uma função
        list, # Tipo na qual a população será inserida 
        toolbox.individual # Função a ser chamada varias vezes
    )

    # Operações
    toolbox.register("select",tools.selTournament,tournsize=3)
    toolbox.register("mate",tools.cxPartialyMatched)
    # Mudar a cidade 
    toolbox.register("mutate",tools.mutShuffleIndexes,indpb=PROBABILIDADE_MUTACAO_INDICE)
    toolbox.register("evaluate",FuncaoCusto)
    return toolbox 


def CriarEstatistica(valorOtimo=None):
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("mean",np.mean)
    stats.register("min",np.min)
    stats.register("std",np.std)
    stats.register("pop",lambda x: x)
    if valorOtimo:
        stats.register("success",lambda x: valorOtimo/np.min(x) > 0.9)
        stats.register("optimal success",lambda x: np.abs(valorOtimo-np.min(x))<1e-8)
        stats.register("gap",lambda x: valorOtimo/np.min(x))
    return stats 

def CriarHallDaFama():
    hof = tools.HallOfFame(5)
    return hof

def GerarFuncaoCustoPermutacao(posicoes):
    def FuncaoCusto(indices):
        return np.sum(np.linalg.norm(posicoes[:,indices]-np.roll(posicoes[:,indices],1,axis=1),axis=0)),
    return FuncaoCusto 

def GerarFuncaoCustoInteiroLimitado(posicoes):
    def FuncaoCusto(indicesCidades):
        # Conjunto de cidades adicionadas
        cidadesAdicionadas = set()
        # novos indices corrigidos
        indicesCidadesCorrigidos = np.arange(len(indicesCidades))
        # Itera por cada cidade do indice 
        for idx, cidade in enumerate(indicesCidades):
            # A cidade atual começa pela cidade do indice 
            cidadeAtual = cidade 
            # Para todos os valores inclusos na lista de cidades 
            for i in range(len(indicesCidades)+1):
                # Se a cidade ainda não foi adicionada à lista
                if not cidadeAtual in cidadesAdicionadas:
                    # A cidade é adicionada ao índice corrigido
                    indicesCidadesCorrigidos[idx] = cidadeAtual
                    # A cidade é adicionada a lista de cidades corrigidas
                    cidadesAdicionadas.add(cidadeAtual)
                    # Sai do loop 
                    break
                # Se a cidade foi adicionada, tenta-se o próximo índice, circulando caso tenha passado
                cidadeAtual += 1
                cidadeAtual %= posicoes.shape[1]
            # Se o loop terminou sem a adição de uma nova cidade, tem algo errado
            if i == len(indicesCidades):
                raise Exception("Comportamento inesperado da função custo")

        if len(cidadesAdicionadas) != posicoes.shape[1]:
            print(cidadesAdicionadas)
            print(indicesCidadesCorrigidos)
            print(indicesCidades)
            print(posicoes.shape)
            raise Exception("Algumas cidades não foram visitadas")

        if len(cidadesAdicionadas.symmetric_difference(set(np.arange(posicoes.shape[1])))) != 0 :
            print(cidadesAdicionadas)
            print(indicesCidadesCorrigidos)
            print(indicesCidades)
            print(posicoes.shape)
            raise Exception("Cidades que não existem foram indexadas")
        # Calcula o resultado com os indices corrigidos 
        return np.sum(np.linalg.norm(posicoes[:,indicesCidadesCorrigidos]-np.roll(posicoes[:,indicesCidadesCorrigidos],1,axis=1),axis=0)),
    return FuncaoCusto 


def ExecutarSGA(toolbox,params,stats:tools.Statistics ):

    NUMERO_GERACOES = params['N']
    PROBABILIDADE_MUTACAO = params['mupb']
    PROBABILIDADE_RECOMBINACAO = params['cxpb'] 
    TAMANHO_INDIVIDUO = params['P']
    TAMANHO_POPULACAO = params['mu']
    PROBABILIDADE_MUTACAO_INDICE = params['muidxpb']

    hof = CriarHallDaFama()
    # Cria indivíduos
    population = toolbox.population(n=TAMANHO_POPULACAO)
    # Cria os logs
    logbook = tools.Logbook()
    
#    __import__('pdb').set_trace()

    # Avaliação inicial
    aptidoes = map(toolbox.evaluate,population)
    for individuo, aptidao in zip(population,aptidoes):
        individuo.fitness.values = aptidao

    record = stats.compile(population)
    logbook.record(gen=1,evals = len(population), **record)
    
    hof.update(population)
    
    notChanged = 0
    lastHof = hof[0].fitness.values[0]

    for i in range(NUMERO_GERACOES):
        offspring = toolbox.select(population,len(population))
        offspring = map(toolbox.clone,offspring)
        offspring = algorithms.varAnd(population,toolbox,PROBABILIDADE_RECOMBINACAO,PROBABILIDADE_MUTACAO)

        # Evaluate the individuals with an invalid fitness
        #invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        #fitnesses = map(toolbox.evaluate, invalid_ind)
        #for ind, fit in zip(invalid_ind, fitnesses):
        #    ind.fitness.values = fit
        # Avaliação inicial
        aptidoes = map(toolbox.evaluate,offspring)
        for individuo, aptidao in zip(offspring,aptidoes):
            individuo.fitness.values = aptidao
        # Geracional 
        population = offspring 
        
        # Salvar estatística
        record = stats.compile(population)
        logbook.record(gen=i+1,evals = len(population), **record)
        
        # Salva Hall of Fame
        hof.update(population)
        if np.abs(lastHof - hof[0].fitness.values[0]) < 1e-10:
            notChanged += 1
            if notChanged == 1000:
                break 
        else:
            notChanged = 0
            lastHof = hof[0].fitness.values[0]

        if record['success'] == True:
            break
        
    return population, logbook, hof

def SGAPermutacao(posicoes,params,vetorOtimo=None):
    # Variáveis necessárias
    TAMANHO_INDIVIDUO = posicoes.shape[1]
    PROBABILIDADE_MUTACAO_INDICE = params['muidxpb']
    
    # Define os valores para o algoritmo SGA
    FuncaoCusto = GerarFuncaoCustoPermutacao(posicoes)
    toolbox = CriarAlgoritmoSGAPermutacao(params['P'],params['muidxpb'],FuncaoCusto)
    pop = toolbox.population(params['mu'])
    hof = tools.HallOfFame(1)
    stats = CriarEstatistica(valorOtimo=FuncaoCusto(vetorOtimo)[0] if not vetorOtimo is None else None)
    pop,log  = algorithms.eaSimple(pop,toolbox,params['cxpb'],params['mupb'],params['N'],stats=stats,verbose=False,halloffame=hof)
    return (pop,log,hof,params,FuncaoCusto)

def SGAInteiroLimitado(posicoes,params,vetorOtimo=None):

    TAMANHO_INDIVIDUO = posicoes.shape[1]
    PROBABILIDADE_MUTACAO_INDICE = params['muidxpb']
    
    # Define os valores para o algoritmo SGA
    FuncaoCusto = GerarFuncaoCustoInteiroLimitado(posicoes)
    toolbox = CriarAlgoritmoSGAInteiroLimitado(TAMANHO_INDIVIDUO,PROBABILIDADE_MUTACAO_INDICE,FuncaoCusto)
    pop = toolbox.population(params['mu'])
    hof = tools.HallOfFame(1)
    stats = CriarEstatistica(valorOtimo=FuncaoCusto(vetorOtimo)[0] if not vetorOtimo is None else None)
    pop,log  = algorithms.eaSimple(pop,toolbox,params['cxpb'],params['mupb'],params['N'],stats=stats,verbose=False,halloffame=hof)
    return (pop,log,hof,params,FuncaoCusto)


def Resultados(pop, logbook, hof, params, FuncaoCusto=None):
     
    gen, avg, min, success, gap, pop = logbook.select('gen','mean','min','success','gap','pop')
    
    print(f"Melhor indivíduo: {hof[0].fitness.values}")
    geracaoEncontrada = success
    print(f"Geração encontrada: {geracaoEncontrada[-1]}")

    fig, ax = plt.subplots()
    ax.plot(gen,avg,label="Média")
    ax.plot(gen,min,label="Menor")
    ax.legend()
    ax.set_title("Progressão estatística da população") 
    ax.set_xlabel("Gerações")
    ax.set_ylabel("Valor")
    plt.savefig("Estatística.png")



    print(np.array(pop).shape)
    for i in range(np.array(pop).shape[1]):
        plt.plot(gen,np.array(pop)[:,i,0])
    plt.title("Aptidão dos indivíduos") 
    plt.xlabel("Gerações")
    plt.ylabel("Valor")
    plt.savefig("Indivídual.png")

if __name__ == "__main__":
    import pandas as pd 
    import matplotlib.pyplot as plt

    import sqlite3

    from generate_tsp import GerarProblemaRadialTSP 
    

    for cidades in [40]:
        print(f"Cidades={cidades}")
        posicoes = GerarProblemaRadialTSP(cidades)
        params = {
            "Problema":"teste",
            "P":posicoes.shape[1],
            "N":1000,
            "mu":50,
            "cxpb":0.7,
            "mupb":0.3,
            "muidxpb":0.01 
        }
        
        #print("SGA Permutação")
        #for i in range (3):
        #    pop, log, hof, params, cost = SGAPermutacao(posicoes,params)
        #    print(f"Feito SGA: hof={hof[0].fitness.values}")
            

        print("SGA Inteiro")
        #for i in range (3):
        #    pop, log, hof, params, cost = SGAInteiroLimitado(posicoes,params)
        #    print(f"Feito SGA: hof={hof[0].fitness.values}")
        #    print(f"Feito SGA: hof={np.unique(np.array(hof))}")

        #Resultados(*SGAPermutacao(posicoes,params))
        Resultados(*SGAInteiroLimitado(posicoes,params))
        
