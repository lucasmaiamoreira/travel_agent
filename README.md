# Agente de Viagem Virtual com Streamlit e Ollama

Este projeto utiliza o Streamlit para criar uma aplicação de agente de viagem virtual que se comunica com o modelo `llama3` através da API do Ollama.

## Pré-requisitos

Antes de começar, verifique se você possui os seguintes pré-requisitos:

- **Python**: Versão 3.10 ou superior
- **pip**: Gerenciador de pacotes do Python
- **curl**: Ferramenta para transferir dados com URL

## Instalação do Ollama

### 1. Instalar Ollama

Para instalar o Ollama, execute o seguinte comando:

```
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Iniciando servidor Ollama:

```
ollama serve 
```

### 3. Baixando modelo llama3 para usar no projeto:

```
ollama run llama3
```

Obs.: Qualquer problema na instalação do Ollama é possível verificar no site oficial da aplicação: https://ollama.com/


#### Instalação e Configuração da Aplicação


1. Clonar o Repositório

Clone este repositório para a sua máquina local:

```
git clone git@github.com:lucasmaiamoreira/travel_agent.git
cd travel_agent
```

2. Instalar Dependências

```
python -m venv venv
source venv/bin/activate  # Para Windows: venv\Scripts\activate
pip install -r requirements.txt
```

#### Executando a Aplicação

Para iniciar a aplicação Streamlit, execute o seguinte comando:

```
streamlit run agente_de_viagem.py
```

#### Autor
Este projeto foi desenvolvido por Lucas Maia Moreira.

#### Contribuições
Contribuições são bem-vindas! Sinta-se à vontade para abrir uma issue ou enviar um pull request.

#### Licença
Este projeto está licenciado sob a Licença MIT.