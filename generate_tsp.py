import numpy as np


def GerarProblemaRadialTSP(numeroCidades,raio=1):
    a = np.linspace(0,2*np.pi,numeroCidades+1)
    return np.stack([raio*np.sin(a),raio*np.cos(a)])


def GerarProblemaRetangularTSP(numeroFocos,cidadesFoco,lado=20):
    a = np.meshgrid(np.arange(numeroFocos),np.arange(numeroFocos))
    a = np.stack(a)*lado 
    a = a.reshape(2,-1)
    a = a.T

    result = None 
    for coordinates in a:
        positions = coordinates[np.newaxis,:]+np.random.normal(0,lado//20,size=(cidadesFoco,2))
        if not type(result) is type(None):
            result = np.concatenate([result,positions])
        else:
            result = positions
    
    return result.T 



if __name__ == "__main__":
    from visualizacoes import PlotResultadoSA
    import matplotlib.pyplot as plt 
    
    tamanho = 10
    posicoesRadial = GerarProblemaRadialTSP(tamanho)
    PlotResultadoSA(np.arange(posicoesRadial.shape[1]), posicoesRadial) 
    plt.title("Problema radial")
    plt.show()

    cidades = 3
    postos = 4
    posicoesRetangular = GerarProblemaRetangularTSP(cidades,postos)
    PlotResultadoSA(np.arange(posicoesRetangular.shape[1]), posicoesRetangular)
    plt.title("Problema retangular")
    plt.show()


    print(posicoesRetangular.shape)
    print(posicoesRadial.shape)
