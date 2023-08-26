#usando o projeto

#windows
#py -3 -m venv venv
venv/Scripts/activate


# linux

#python3 -m venv venv
#source venv/bin/activate


pip install -r requirements.txt

#usar um csv no formato padrao

#executar o notebook



MLPRegressor

Hidden_layer_sizes semelhante a uma matriz de forma (n_layers - 2,), padrão = (100,)
O i-ésimo elemento representa o número de neurônios na i-ésima camada oculta.

ativação {'identidade', 'logística', 'tanh', 'relu'}, padrão='relu'
Função de ativação da camada oculta.

'identidade', ativação no-op, útil para implementar gargalo linear, retorna f(x) = x

'logística', a função sigmóide logística, retorna f(x) = 1/(1 + exp(-x)).

'tanh', a função tan hiperbólica, retorna f(x) = tanh(x).

'relu', a função de unidade linear retificada, retorna f(x) = max(0, x)

solucionador {'lbfgs', 'sgd', 'adam'}, padrão='adam'
O solucionador para otimização de peso.

'lbfgs' é um otimizador da família de métodos quase-Newton.

'sgd' refere-se à descida gradiente estocástica.

'adam' refere-se a um otimizador estocástico baseado em gradiente proposto por Kingma, Diederik e Jimmy Ba

Observação: o solucionador padrão 'adam' funciona muito bem em conjuntos de dados relativamente grandes (com milhares de amostras de treinamento ou mais) em termos de tempo de treinamento e pontuação de validação. Para conjuntos de dados pequenos, entretanto, 'lbfgs' pode convergir mais rapidamente e ter melhor desempenho.

alfa flutuante, padrão = 0,0001
Força do termo de regularização L2. O termo de regularização L2 é dividido pelo tamanho da amostra quando somado à perda.

batch_size int, padrão='auto'
Tamanho dos minilotes para otimizadores estocásticos. Se o solucionador for 'lbfgs', o regressor não usará minibatch. Quando definido como “automático”, .batch_size=min(200, n_samples)

taxa_de_aprendizado {'constante', 'invscaling', 'adaptável'}, padrão='constante'
Cronograma de taxa de aprendizagem para atualizações de peso.

'constante' é uma taxa de aprendizagem constante dada por 'learning_rate_init'.

'invscaling' diminui gradualmente a taxa de aprendizagem learning_rate_ em cada passo de tempo 't' usando um expoente de escala inverso de 'power_t'. taxa_de_aprendizagem_efetiva = taxa_de_aprendizado_init / pow(t, potência_t)

'adaptativo' mantém a taxa de aprendizagem constante em 'learning_rate_init' enquanto a perda de treinamento continuar diminuindo. Cada vez que duas épocas consecutivas não conseguem diminuir a perda de treinamento em pelo menos tol, ou não conseguem aumentar a pontuação de validação em pelo menos tol se 'early_stopping' estiver ativado, a taxa de aprendizado atual é dividida por 5.

Usado apenas quando solucionador='sgd'.

learning_rate_init flutuante, padrão = 0,001
A taxa de aprendizagem inicial usada. Ele controla o tamanho do passo na atualização dos pesos. Usado apenas quando solver='sgd' ou 'adam'.

power_t float, padrão = 0,5
O expoente da taxa de aprendizagem de escala inversa. É usado na atualização da taxa de aprendizagem efetiva quando learning_rate está definido como 'invscaling'. Usado apenas quando solucionador='sgd'.

max_iter int, padrão=200
Número máximo de iterações. O solucionador itera até a convergência (determinada por 'tol') ou este número de iterações. Para solucionadores estocásticos ('sgd', 'adam'), observe que isso determina o número de épocas (quantas vezes cada ponto de dados será usado), não o número de etapas do gradiente.

embaralhar bool, padrão = Verdadeiro
Se as amostras devem ser embaralhadas em cada iteração. Usado apenas quando solver='sgd' ou 'adam'.

random_state int, instância RandomState, padrão = Nenhum
Determina a geração de números aleatórios para pesos e inicialização de polarização, divisão de teste de trem se a parada antecipada for usada e amostragem em lote quando solucionador='sgd' ou 'adam'. Passe um int para resultados reproduzíveis em múltiplas chamadas de função. Consulte Glossário .

tol float, padrão = 1e-4
Tolerância para a otimização. Quando a perda ou pontuação não melhora pelo menos tolem n_iter_no_changeiterações consecutivas, a menos que learning_rateseja definido como 'adaptativo', a convergência é considerada alcançada e o treinamento é interrompido.

bool detalhado , padrão=Falso
Se deve imprimir mensagens de progresso em stdout.

Warm_start bool, padrão=Falso
Quando definido como True, reutiliza a solução da chamada anterior para caber como inicialização, caso contrário, apenas apague a solução anterior. Consulte o Glossário .

flutuação de impulso , padrão = 0,9
Momentum para atualização de descida gradiente. Deve estar entre 0 e 1. Usado somente quando solver='sgd'.

nesterovs_momentum bool, padrão = Verdadeiro
Seja para usar o impulso de Nesterov. Usado apenas quando solver='sgd' e momentum > 0.

parada_inicial bool, padrão=Falso
Se deve-se usar a parada antecipada para encerrar o treinamento quando a pontuação de validação não estiver melhorando. Se definido como True, ele deixará automaticamente de lado os validation_fractiondados de treinamento como validação e encerrará o treinamento quando a pontuação de validação não estiver melhorando, pelo menos tolem n_iter_no_changeépocas consecutivas. Efetivo apenas quando solver='sgd' ou 'adam'.

validação_fração flutuante, padrão = 0,1
A proporção de dados de treinamento a serem reservados como conjunto de validação para parada antecipada. Deve estar entre 0 e 1. Usado somente se early_stopping for True.

beta_1 flutuante, padrão = 0,9
A taxa de decaimento exponencial para estimativas do primeiro vetor de momento em Adam deve estar em [0, 1). Usado apenas quando solucionador='adam'.

beta_2 flutuante, padrão = 0,999
A taxa de decaimento exponencial para estimativas do segundo vetor de momento em Adam deve estar em [0, 1). Usado apenas quando solucionador='adam'.

flutuação épsilon , padrão = 1e-8
Valor para estabilidade numérica em Adam. Usado apenas quando solucionador='adam'.

n_iter_no_change int, padrão=10
Número máximo de épocas para não atender tolà melhoria. Efetivo apenas quando solver='sgd' ou 'adam'.

Novo na versão 0.20.

max_fun int, padrão=15000
Usado apenas quando solucionador='lbfgs'. Número máximo de chamadas de função. O solucionador itera até a convergência (determinada por tol), o número de iterações atinge max_iter ou esse número de chamadas de função. Observe que o número de chamadas de função será maior ou igual ao número de iterações do MLPRegressor.