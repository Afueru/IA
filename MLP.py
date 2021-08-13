"""NOMES E NÚMEROS USP:
Gabriel Assenço Ferreira de Siqueira - NºUSP: 11295887 
Guilherme Kenzo Silva Oshiro - NºUSP: 11314988
Guilherme Rodrigues Pisni - NºUSP: 11270851
Icaro Chellucci Macedo dos Santos - NºUSP: 11270973 
Mark Poll Herrmann - NºUSP: 11208291
"""
import numpy as np
from numpy.core.fromnumeric import around
import pandas
import os

Erros_medios = []

class MLP:
	def __init__(self, n_entrada = 2,n_escondida = [3], n_saida = 1, bias = False):

		""" 
		Construtor do MLP
		n_entrada: numero de neurônios de entrada
		n_escondida: array que define o formato da camada escondida, cada elemento do array é a quantidade de neurônios por camada escondida, portanto, o número de camadas escondidas é definido pela quantidade de elementos no array
		n_saida: número de neurônios de saída
		bias: boolean que indica se haverá bias ou não
		"""

		self.n_entrada = n_entrada			
		self.n_escondida = n_escondida
		self.n_saida = n_saida							#seta os parâmetros passados para a MLP

		self.lay = [self.n_entrada] + n_escondida + [n_saida]		#array que salva a quantidade de neuronios por camada


		self.pesos = []
		for i in range(len(self.lay) - 1):								#inicializa os pesos aleatoriamente em matrizes de tamanho = (n_linhas = tamanho_camada_atual, n_colunas = tamanho_camada_seguinte)
			p = np.random.uniform(-1,1,(self.lay[i],self.lay[i+1]))
			self.pesos.append(p)

		
		self.neuronios = []												#lista com os resultados de cada neuronio (inicialmente 0 para todos)
		for i in range(len(self.lay)):
			neu = np.zeros(self.lay[i])
			self.neuronios.append(neu)

		self.bias = []
		if (bias == True):												#inicializa o Bias aleatoriamente se o boolean for True
			for i in range(len(self.lay) - 1):
				b = np.random.uniform(-0.5,0.5,(len(self.neuronios[i + 1]))) # ,3 // arr = [[0.2,0.3]]
				self.bias.append(b)											
			self.d_bias = []											#inicializa o array de deltas dos bias
			for i in range(len(self.bias)):
				db = np.zeros(len(self.bias[i]))
				self.d_bias.append(db)


		else:														#inicializa o Bias com valores iguais a zero se o boolean for false, o resultado prático é não haver bias					
			for i in range(len(self.lay) - 1):
				b = np.zeros(len(self.neuronios[i + 1]))
				self.bias.append(b)
			self.d_bias = False										#boolean para garantir que não há bias
		self.exportPesos("pesosIniciais.txt")	
		self.deltas = []											#lista que conterá os deltas para a correção dos erros, com o mesmo tamanho da camada atual de neuronios (inicialmente 0 para todos)
		for i in range(len(self.lay) - 1):
			der = np.zeros((self.lay[i],self.lay[i + 1]))
			self.deltas.append(der)

		self.confusao = np.zeros((n_saida, n_saida))				#matriz de confusão

	def separa_dados (self, linha, resposta = False):														#Função para separar entrada da saída em uma linha, recebe a linha e o boolean para verificar se quer o conjunto de respostas ou de entradas
		if (resposta == True):
			dados = [linha[j] for j in range(len(linha) - self.n_saida,len(linha))]
		else:
			dados = [linha[j] for j in range(len(linha) - self.n_saida)]
		return dados


	def resposta (self, adq, arredondado = True, caracter = False):											#função para retornar a resposta da rede neural
		resposta = np.zeros(len(self.neuronios[-1]))														#recebe adq(o conjunto para arredondar os resultados), boolean para saber se o resultado deve ser arredondado e boolean para saber se quer o resultado em caracteres ou não
		if (arredondado == True):																			#se quiser arredondado, ele apenas arredonda os resultados adquiridos e retorna num array
			for i,resp in enumerate(adq):
				resposta[i] = np.around(resp)
		else:
			resposta = adq

		if (caracter == True):																				#se quiser em caracteres, a função concatenará os caracteres que a rede encontrar como resultados e retornará a variável
			resp_caracter = ''
			if (resposta[0] == 1):
				resp_caracter += 'A'
			if (resposta[1] == 1):
				resp_caracter += 'B'
			if (resposta[2] == 1):
				resp_caracter += 'C'
			if (resposta[3] == 1):
				resp_caracter += 'D'
			if (resposta[4] == 1):
				resp_caracter += 'E'
			if (resposta[5] == 1):
				resp_caracter += 'J'
			if (resposta[6] == 1):
				resp_caracter += 'K'
			"""if (resp_caracter == ''):																		#se no fim, a rede não chegar numa resposta, retorna os resultados da camada arredondados em 3 casas decimais
				for i, resp in enumerate(adq):
					resposta[i] = np.around(resp, decimals = 3)
				return np.asarray(resposta).tolist()"""														#desativado por poluir demais o console na saída

			return resp_caracter

				
			return resposta

	def sigmoid (self,x):
		return (1.0/(1 + np.exp(-x)))							#função de ativação: sigmoide

	def derivada_sig (self,x):
		return x * (1 - x)										#derivada da função de ativação

	def treinar (self, raw_dados, alfa = 0.5, epocas = 1):													#função de treinamento, recebe os dados brutos para o treinamento, coeficiente de aprendizado (alfa) e o número de epocas
		
		rng = np.random.default_rng()																		#criação de uma classe "random generator" da biblioteca numpy, que servirá para gerar números aleatórios
		random_index = rng.choice(len(raw_dados) - 1, size= int(np.around((len(raw_dados)/4))), replace = False)		#escolhe aleatoriamente n números para o conjunto de validação e cria um array contendo eles
		valida = raw_dados[random_index,:]																	#n vai cerca de 1/4 do tamanho dos dados para treinamento
		dados = np.delete(raw_dados, random_index, axis = 0)												#cria um conjunto de dados sem os dados do conjunto de validação
		index_rest = np.delete(np.asarray([i for i in range(len(raw_dados))]),random_index,axis = 0)		#inicializa array que mostrará as linhas do arquivo de dados que serão utilizadas para o treinamento
		index_rest = index_rest + 1 																		#soma 1 para ter as linhas de fato, já que o índice corresponderia à (n_linha - 1)
		random_index = np.asarray(random_index) + 1
		print("Conjunto de treinamento: " + str(index_rest))
		print("Conjunto de validacao: " + str(random_index))

		results = [0 for i in range(len(dados))]															#lista que armazenará os resultados finais

		max_erros = 50																						#máximo aceitável de erros seguidos
		val_erro = 1 																						#variável que armazenará o erro do conjunto de validação da epoca atual
		val_counter = 0 																					#contador de erros crescentes seguidos do conjunto de validação
		ant_erro = 1 																						#erro do conjunto de validação da época anterior
		erro_medio = 1 																						#variável que armazena o erro do treinamento atual
		ant_erro_medio = 1 																					#variável que armazena o erro do treinamento anterior
		vai = True																							#boolean para definir o ponto da parada antecipada


		for epoca in range(epocas):
			erros_medios = []
			for index,linha in enumerate(dados):
				inputs = self.separa_dados(linha)															#separa as entradas das respostas esperadas
				resp = self.separa_dados(linha, resposta = True)
				self.feedforward(inputs)																	#roda o feedforward para prever o resultado

				erro = np.subtract(resp, self.neuronios[-1])												#define o erro da rede no conjunto atual
				erro_medio_it = np.average(erro**2)															#define o erro médio dos neurônios nessa iteração
				erros_medios.append(erro_medio_it)															#adiciona à um array para gerenciar os erros nessa época

				self.backpropagation(resp, erro)															#backpropagation para definir os deltas

				self.atualiza_pesos(alfa)																	#atualiza os pesos com base nos deltas
				#print(index)
				#results[index] = np.around(self.neuronios[-1], decimals = 1)
				arr = []
				

				for res in self.neuronios[-1]:																#essa parte serve apenas para fomatar os números adequadamente para o print
					"""if (res>=0.9 or res <= 0.1):
						arr.append(np.around(np.around(res, decimals = 1)))
					else:
						arr.append(np.around(res, decimals = 1))"""
					arr.append(np.around(res, decimals= 3))
				
				results[index] = arr 																		#armazena os resultados dessa iteração

			ant_erro = val_erro
			ant_erro_medio = erro_medio																		#redefine o erro anterior
			erro_medio = np.average(erros_medios)															#faz a média dos erros de cada iteração na época e armazena em uma variável
			saida("Erro.txt",erro_medio, epoca)																#imprime o erro médio no arquivo definido
			val_erro = self.valida(valida, val_erro)														#verifica os erros do conjunto de validação e retorna seu erro nessa variável
			saida("Val_Erro.txt",val_erro, epoca)															#imprime o erro do conjunto de validação no arquivo definido

			if (val_counter <= max_erros):																	#se o contador for menor que o máximo de erros seguidos
				saida("Erro_PA.txt", erro_medio, epoca)														#imprime nos arquivos de parada antecipada
				saida("Val_Erro_PA.txt", val_erro, epoca)
				results_pa = results.copy()																	#copia o array de resultados para gerenciamento
			if ((vai == True and val_counter > max_erros) or (epoca == epocas - 1 and vai == True)):		#se o contador passar dos erros máximos ou for a ultima época										
				self.exportPesos("PA_Pesos.txt")															#exporta os pesos atuais
				vai = False																					#define o boolean como falso para não resetar o contador

			if (val_erro > ant_erro and ant_erro_medio > erro_medio and epoca >= (8/100)*epocas):			#se o erro atual for maior que o erro anterior (no caso da validação). Também deve ser satisfeita a condição do erro medio anterior ser maior que o erro medio atual do treinamento
				val_counter += 1 																			#soma 1 no contador
				#print("anterior {} atual {}".format(ant_erro, val_erro))
				#if (val_counter >= max_erros):
					#break
			else:																							#se não, reseta o contador
				if (vai == True):
					val_counter = 0


			#print("Epoca " + str(epoca + 1) + " concluida")
		"""print("--------------------- RESULTADOS APÓS TREINAMENTO COM PARADA ANTECIPADA ---------------------")
		for i in range(len(results_pa)):
			linha = dados[i]
			resp = [linha[j] for j in range((len(linha) - self.n_saida),len(linha))]
			print("Resposta esperada na linha: " + str(i + 1) + ": " + str(resp)) 
			print("Resultados linha " + str(i+1) + ": " + str(results_pa[i]) + "\n")

		print("--------------------- RESULTADOS APÓS TREINAMENTO SEM PARADA ANTECIPADA ---------------------")
		for i in range(len(results)):
			linha = dados[i]
			resp = [linha[j] for j in range((len(linha) - self.n_saida),len(linha))]
			print("Resposta esperada na linha: " + str(i + 1) + ": " + str(resp)) 
			print("Resultados linha " + str(i+1) + ": " + str(results[i]) + "\n")"""

	def valida (self,dados, ant_erro):																		#Função para fazer a validação dos dados, recebe o conjunto de validação e o erro anterior
		val_erros = []
		for i,linha in enumerate(dados):
			inputs = self.separa_dados(linha)																	#separa inputs das respostas esperadas
			resp = self.separa_dados(linha, resposta = True)									
			self.feedforward(inputs)																		#faz previsão dos resultados

			erro = np.subtract(resp, self.neuronios[-1])													#calcula os erros médios de cada iteração e adiciona em um array
			val_erro_it = np.average(erro**2)
			val_erros.append(val_erro_it)

		val_erro = np.average(val_erros)																	#define o erro da validação como a média dos erros de cada iteração
		return val_erro 																					#retorna o erro da validação
		
	def projeta (self, dados):																				#função para projetar o resultado passado
		results = []																						#array que armazenará os resultados da rede

		for linha in dados:
			inputs = self.separa_dados(linha)
			resp = self.separa_dados(linha, resposta = True)
			self.feedforward(inputs)																		#faz o feedforward para prever o resultado da rede
			results.append(self.resposta(self.neuronios[-1],caracter = True))								#armazena o resultado (em formato de caracter) da rede em um array
			self.adicionaConfusao(self.neuronios[-1], resp)													#adiciona à matriz de confusão
		


		return results  																					#retorna os resultados

	def adicionaConfusao (self, adq, resp):																	#função para adicionar à matriz de confusão, recebe conjunto de respostas adquiridas e respostas esperadas
		indexes_adq = []																					#array que armazena os indexes em que os neuronios arredondados forem iguais a 1
		indexes_resp = []																					#array que armazena o index do array de resposta que contém o valor 1
		for i,num in enumerate(adq):																		#passará pelas respostas adquiridas e adicionará à um array de indexes, todos os neurônios arredondados que forem iguais a 1																			
			if (np.around(num) == 1):
				indexes_adq.append(i)
			
			if (resp[i] == 1):																				#checa também qual é o neurônio correto de saída no conjunto de respostas
				indexes_resp.append(i)			

		for i in (range(len(indexes_adq))):																	#adiciona à matriz de confusão
			self.confusao[indexes_resp[0]][indexes_adq[i]] += 1



		
	def feedforward (self, inputs):
		self.neuronios[0] = inputs			#a primeira camada é a camada de entrada			
		pos_ativ = []						#array que armazenará o resultado do calculo da camada

		for i in range(len(self.pesos)):						#para cada array de pesos, será feita a multiplicação de matrizes entre a saída da camada anterior e a camada de pesos atual
			prod = np.dot(self.neuronios[i],self.pesos[i])
			prod_bias = prod + self.bias[i]						#soma o bias, se o bias for 0, a soma permanecerá intacta
			pos_ativ = self.sigmoid(prod_bias)					#passa os valores da multiplicação das matrizes para a função de ativação
			self.neuronios[i+1] = pos_ativ						#armazena a matriz contendo os resultados da ativação como saída da camada seguinte



	def backpropagation (self, resp, erro):
		#calcular: erro = (resposta_esperada - resposta_obtida) = resp - self.neuronios[-1] 
		#delta(in_j) = erro * derivada_sig(neuronios[i + 1])
		#delta_peso = delta(in_j)*neuronios[i] (saídas do neurônio anterior = entrada do peso atual)
		#erro seguinte: e = delta(in_j) ● pesos[i].T




		for i in reversed(range(len(self.deltas))): 											#for que rodará de trás para frente, tendo com base o tamanho da matriz de deltas
			output = self.neuronios[i + 1]
			output_derivada = self.derivada_sig(output)
			delta = erro * output_derivada														#delta terá a mesma dimensão da camada de neuronios posterior
			delta_ajustado = delta.reshape(delta.shape[0],-1).T 								#queremos uma matriz com as mesmas dimensões do peso atual
			atual = np.asarray(self.neuronios[i])												#para isso precisamos multiplicar as matrizes, redimensionando elas para resultar na matriz desejada
			atual_ajustado = atual.reshape(atual.shape[0],-1)									#assim, supondo que a matriz peso tenha dimensões (3,2), teremos uma multiplicação entre as matrizes:
			self.deltas[i] = np.dot(atual_ajustado,delta_ajustado)								#[[a1][a2][a3]] ● [[d1, d2]] = [[a1*d1,a1*d2] [a2*d1,a2*d2] [a3*d1,a3*d2]]
			erro = np.dot(delta, self.pesos[i].T)												#para obtermos o erro da camada anterior, devemos pegar o erro atual e multiplicar pelos pesos, para isso a matriz de pesos deve ser transposta

		if (self.d_bias != False):
			erro = np.subtract(resp, self.neuronios[-1])

			for i in reversed(range(len(self.bias))):											#para ajustar o bias, basta recriarmos o delta, como era calculado no passo anterior 
				output = self.neuronios[i + 1]													
				output_derivada = self.derivada_sig(output)
				delta = erro * output_derivada
				self.d_bias[i] = delta 															#então, armazenamos no array para depois ajustar o bias
				erro = np.dot(delta, self.pesos[i].T)
				
	def atualiza_pesos(self, alfa):
		for i in range(len(self.neuronios) - 1):
			self.pesos[i] = self.pesos[i] + (alfa * self.deltas[i])		#mork					#atualiza o peso somando o peso atual com uma multiplicação entre os deltas calculados e o coeficiente de aprendizado
			
		if (self.d_bias != False):																#checa se existe bias
			for i in range(len(self.bias)):
				self.bias[i] = self.bias[i] + (alfa * self.d_bias[i])							#atualiza o bias somando o bias atual com uma multiplicação entre os deltas dos bias e o coeficiente de aprendizado

	def exportPesos(self, file):																#função para exportar os pesos no arquivo especificado
		filewriter = open(file, "w")
		for matriz in self.pesos:																#irá escrever cada linha de cada matriz em uma linha no arquivo especificado
			for linha in matriz:
				for num in linha:
					filewriter.write("{} ".format(num))
				filewriter.write("\n")
		for linha in self.bias:																	#faz o mesmo processo com os bias
			for num in linha:
				filewriter.write("{} ".format(num))
			filewriter.write("\n")

	def loadPesos(self, file):																	#função que carrega os pesos e bias baseados em um arquivo feito pela função "exportPesos"
		file_reader = open(file, "r")															#IMPORTANTE: O MLP QUE CARREGA OS PESOS DEVE TER AS MESMAS DIMENSÕES POR CAMADA DO MLP QUE EXPORTOU OS PESOS
		arquivo = file_reader.readlines()	
		m = []
		counter = 0
		for i,matriz in enumerate(self.pesos):													#lê cada linha e armazena em uma matriz "m"
			temp = np.zeros((self.lay[i],self.lay[i+1]))
			for j in range(len(matriz)):
				linha = arquivo[counter].split()
				for k,num in enumerate(linha):
					temp[j][k] = float(num)
				counter += 1 																	#ao final de cada linha, indica ao contador para que vá para a próxima linha
			m.append(temp)
		self.pesos = m 																			#define a matriz "m" como matriz de pesos
		b = []
		for camada in self.bias:																#faz o mesmo processo para os bias
			linha = arquivo[counter].split()
			temp = np.zeros(len(camada))
			for i,num in enumerate(linha):
				temp[i] = float(num)
			counter += 1
			b.append(temp)
		self.bias = b

	def reiniciaMatriz(self):
		self.confusao = np.zeros((n_saida, n_saida))


def exportResultados(file, origem, resultados,first = False):									#função para exportar os resultados em um arquivo
	if (first == True):
		file_writer = open(file, "w")
	else:
		file_writer = open(file, "a")
	
	file_writer.write("Arquivo: " + str(origem) + "\n")
	file_writer.write(str(resultados) + "\n")
	file_writer.write("\n")



						
def porcentagem_acerto (resultados, esperado):
	acertos = 0
	for i,r in enumerate(resultados):
		if (r == esperado[i]):
			acertos += 1
	return (int((acertos/len(esperado)*100)))

def saida(file, erro_medio, tentativa):
	if (tentativa == 1): 
		Erro_File = open(file,"w")
	else:
		 Erro_File = open(file,"a")

	#Erro_File.write("erro na tentativa {}: {}\n".format(tentativa, erro_medio))
	Erro_File.write("{} {}\n".format(tentativa, erro_medio))

	Erro_File.close

def nomes_numeros_usp ():
	integrantes = ["Gabriel Assenço Ferreira de Siqueira - NºUSP: 11295887","Guilherme Kenzo Silva Oshiro - NºUSP: 11314988","Guilherme Rodrigues Pisni - NºUSP: 11270851","Icaro Chellucci Macedo dos Santos - NºUSP: 11270973","Mark Poll Herrmann - NºUSP: 11208291"]
	file_writer = open ("NOMES E NUMEROS USP.txt", "w")
	file_writer.write("")
	for integrante in integrantes:
		file_writer.write(integrante + "\n")




#main começa aqui
np.seterr(all='warn')
nomes_numeros_usp()
dir = os.getcwd() + '\\Samples\\'
file = 'caracteres-limpo.csv'
#raw = pandas.read_csv((os.getcwd() + dir + file),skiprows=[0],header = None)
raw = pandas.read_csv((dir + file),header = None)															#linha para ler o csv
dados = np.asarray(raw)

n_saida = 7
n_entrada = len(dados[0]) - n_saida
hid = [3]																									#camadas escondidas [2,3] --> 2 camadas, uma com 2 neuronios e outra com 3
mlp = MLP(n_entrada, hid, n_saida, bias = True)

print("---------------TREINAMENTO---------------")
mlp.treinar(dados,0.1,20000)
mlp.exportPesos("pesosFinais.txt")
print("-----------FIM DO TREINAMENTO------------")
print("")

file_nomes = [file] + ['caracteres-ruido.csv'] + ['caracteres_ruido20.csv']
files = [raw] + [pandas.read_csv((dir + 'caracteres-ruido.csv'),header = None)] + [pandas.read_csv((dir +'caracteres_ruido20.csv'),header = None)]

print("--------------------------TESTES SEM PARADA ANTECIPADA--------------------------")
for i,f in enumerate(files):

	dados = np.asarray(f)	
	resultados = mlp.projeta(dados)
	esperado = []
	for linha in dados:
		esp = mlp.resposta(mlp.separa_dados(linha, resposta = True),caracter = True)
		esperado.append(esp)
	if (i == 0):
		exportResultados("Resultados MLP.txt", file_nomes[i], resultados, True)
	else:
		exportResultados("Resultados MLP.txt", file_nomes[i], resultados)

	print("Resultados obtidos: " + str(resultados))
	print("Resultados esperados: " + str(esperado))
	print("Matriz de confusao: \n" + str(mlp.confusao) + "\n")
	print("Porcentagem de acerto: " + str(porcentagem_acerto(resultados,esperado)) + "%\n")

print("")
print("----------------------FIM DOS TESTES SEM PARADA ANTECIPADA----------------------")
print("")
print("--------------------------TESTES COM PARADA ANTECIPADA--------------------------")
mlp.loadPesos("PA_Pesos.txt")
mlp.reiniciaMatriz()

for i,f in enumerate(files):

	dados = np.asarray(f)	
	resultados = mlp.projeta(dados)
	esperado = []
	for linha in dados:
		esp = mlp.resposta(mlp.separa_dados(linha, resposta = True),caracter = True)
		esperado.append(esp)
	
	if (i == 0):
		exportResultados("Resultados MLP_PA.txt", file_nomes[i], resultados, True)
	else:
		exportResultados("Resultados MLP_PA.txt", file_nomes[i], resultados)


	print("Resultados obtidos: " + str(resultados))
	print("Resultados esperados: " + str(esperado))
	print("Matriz de confusao: \n" + str(mlp.confusao) + "\n")
	print("Porcentagem de acerto: " + str(porcentagem_acerto(resultados,esperado)) + "%\n")

print("")
print("----------------------FIM DOS TESTES COM PARADA ANTECIPADA----------------------")
print("")

#saida()






#print("Neuronios: " + str(mlp.neuronios))