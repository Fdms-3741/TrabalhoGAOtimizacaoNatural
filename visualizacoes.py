import os
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


def CriarFig(fig,ax):
    if fig is None:
        fig,ax = plt.subplots()
    elif ax is None:
        raise Exception("Tem que passar fig e ax juntos")
    return fig,ax

def PlotPontosTSP(posicoes,fig=None,ax=None):
    
    fig,ax = CriarFig(fig,ax)

    ax.scatter(posicoes[0,:],posicoes[1,:])

    return fig, ax

def PlotResultado(x,posicoes,fig=None,ax=None):
    fig,ax = CriarFig(fig,ax)
    
    # Adiciona o primeiro elemento no fim do vetor 
    indices = np.concatenate([x,[x[0]]])
    dados = posicoes[:,indices]
    ax.plot(dados[0,:],dados[1,:], '.r-',markersize=20) 
    
    return fig,ax

sci = lambda x: f'{x/(np.floor(np.log10(x)) if np.floor(np.log10(x))!= 0 else 1):.3f}*10^({np.floor(np.log10(x))})'

if __name__ == "__main__":
    import pandas as pd 
    import sys
    from pathlib import Path

    caminhoArquivo = Path(sys.argv[1])
    if caminhoArquivo.suffix != ".pickle":
        raise Exception("Apenas se processa arquivos .pickle")
    if caminhoArquivo.parent.name != 'resultados':
        raise Exception("Arquivo deve estar em um diretório 'resultados'")

    resultados = pd.read_pickle(caminhoArquivo)
    
    
    for indiceResultados,(key, item) in enumerate(resultados.T.items()):
        saida = (caminhoArquivo.parent / caminhoArquivo.stem / (caminhoArquivo.stem+f'_{indiceResultados}'))
        
        if not saida.exists():
            print("Arquivo de saída inexistente, criando novo")
            saida.mkdir(parents=True)
        elif not saida.is_dir():
            print(f"Arquivo '{saida}' existe e não é um diretório")
            quit(1)
        
        # Apaga arquivos dentro de saida
        arquivosApagados = False
        for arquivo in saida.iterdir():
            arquivosApagados = True
            arquivo.unlink()
        
        if arquivosApagados:
            print(f"Arquivos apagados dentro de '{saida}'")

        figIdx = 1
        # Resultados numéricos simples
        resultadosNumericos = f"K: "+str(item['K'])+"\n"\
            +f"T: "+str(item['$T_0$'])+"\n"\
            +f"N: "+str(item['N'])+"\n"\
            +f"e: "+str(item['$\epsilon$'])+"\n"\
            +f"Tempo de execução: {sci(item['Tempo total'])}\n"\
            +f"Tipo de parada: {item['Condição de parada']}\n"
        
        fig, ax = plt.subplots()
        ax.text(0,0.5,resultadosNumericos,fontsize=18,ha='left')
        ax.axis('off')
        plt.savefig(saida/f'{indiceResultados:03d} - 000 - Resultados numéricos.png')
        plt.close()

        # Grafico de posições puras
        posicoes = item['Posições']
        epsilon = item['$\epsilon$']
        params = f" (K={item['K']}; N={item['N']}; $T_0$={item['$T_0$']};" + " $\epsilon$=" + f'{epsilon})'
        PlotPontosTSP(posicoes)
        plt.title("Posições das cidades")
        plt.savefig(saida/f"{indiceResultados:03d} - {figIdx:03d} - Posições.png")
        figIdx += 1
        plt.close()
        # Gráfico da solução inicial
        PlotResultadoSA(item['$X_0$'],posicoes)
        plt.title(f"Resultado inicial (J={item['$J_0$']})"+params)
        plt.savefig(saida/f"{indiceResultados:03d} - {figIdx:03d} - Resultado inicial.png")
        figIdx += 1
        plt.close()
        # Grafico da solução final
        PlotResultadoSA(item['X'],posicoes)
        plt.title(f"Resultado final (J={item['J']})"+params)
        plt.savefig(saida/f"{indiceResultados:03d} - {figIdx:03d} - Resultado final.png")
        figIdx += 1
        plt.close() 

        # Plot da evolução ao longo do algoritmo
        for nome in ['Evolução J','Evolução T','Evolução $J_{min}$']:
            fig,ax = plt.subplots()
            ax.plot(item[nome])
            ax.set_title(nome + params)
            fig.savefig(saida/(f"{indiceResultados:03d} - {figIdx:03d} - "+nome+".png"))
            figIdx += 1
            plt.close() 
        
        # Plot linear das transições 
        transicoes = item['Transições'] # T, jmin, jx, % aceitacao, % custo, %pertubacai
        for idx, nome in enumerate(['Transições de $T$','Transições de $J_{min}$','Transições de $J_x$','Razão de aceitação','Tempo médio de custo','Tempo médio de pertubação']):
            fig,ax = plt.subplots()
            ax.plot(transicoes[:,idx])
            ax.set_xlabel('$K$')
            ax.set_title(nome + f'(mean: {np.mean(transicoes[:,idx]):.3f}, std: {np.std(transicoes[:,idx]):.3f})')
            fig.savefig(saida/(f"{indiceResultados:03d} - {figIdx:03d} - "+nome+'.png'))
            figIdx += 1
            plt.close()

        # Plot das transições das respostas
        transicoesX = item['Transições X']
        for idx in range(0,transicoes.shape[0],transicoes.shape[0]//5):
            PlotResultadoSA(transicoesX[idx].astype(np.int64),posicoes)
            plt.title(f"Estado da solução em $T={transicoes[idx,0]:.2f}$ ($J={transicoes[idx,1]:.3f}$)")
            plt.savefig(saida/f"{indiceResultados:03d} - {figIdx:03d} - Estado no passo {idx}.png")
            figIdx += 1
            plt.close() 
        
