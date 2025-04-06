# Integrantes
Daniel Spiewak Rempel Finimundi

Luis Felipe Brentegani

# Exportação de Modelo GPT-2 para TFLite com Suporte a LoRA e Quantização

Este repositório contém um código para exportar um modelo GPT-2 treinado (ou adaptado) para o formato TFLite, com suporte para **LoRA (Low-Rank Adaptation)** e **quantização** opcional. A exportação para TFLite permite que o modelo seja executado de maneira eficiente em dispositivos móveis e embarcados, com redução de tamanho e melhoria no desempenho.

## Descrição

O código exporta um modelo GPT-2 para o formato **TFLite**, oferecendo flexibilidade para aplicar **LoRA** nas camadas de atenção e aplicar **quantização** para otimização. O processo de exportação envolve os seguintes passos principais:

1. **Criação e adaptação do modelo GPT-2**:
    - O modelo é carregado a partir de um pré-modelo GPT-2 utilizando o `GPT2Generator`, com a possibilidade de aplicar LoRA e configurar a taxa de aprendizado.
    
2. **Exportação do modelo para o formato SavedModel**:
    - O modelo é exportado para o formato TensorFlow **SavedModel**, incluindo assinaturas para treinamento, inferência e obtenção de pesos.

3. **Conversão para TFLite**:
    - O modelo é então convertido para o formato **TFLite** utilizando o `TFLiteConverter`, com a opção de aplicar **quantização** para otimização.

4. **Salvamento do modelo TFLite**:
    - O modelo convertido é salvo em um arquivo binário no formato TFLite, que pode ser carregado e executado em dispositivos móveis.

## Requisitos

Para rodar este código, você precisará dos seguintes pacotes Python:

- `tensorflow` (para o modelo GPT-2 e conversão para TFLite)
- `transformers` (para carregar o modelo GPT-2 pré-treinado e o tokenizer)
- `argparse` (para manipulação de argumentos de linha de comando)

Você pode instalar as dependências com:

# Como usar

## 1. Configuração dos Argumentos
O código aceita parâmetros de linha de comando para controlar o comportamento do processo de exportação. Os principais parâmetros são:

- `model_name`: Nome do modelo pré-treinado GPT-2 (ex: "gpt2").
- `learning_rate`: Taxa de aprendizado para o otimizador.
- `apply_lora`: Se True, aplica a adaptação LoRA nas camadas de atenção.
- `quantization`: Se True, aplica quantização no modelo para reduzir seu tamanho e melhorar o desempenho em dispositivos móveis.
- `filename`: Nome base do arquivo de saída (ex: "model").
- `export_dir`: Diretório onde o modelo SavedModel será salvo.

## 2. Comandos de Linha
Use o seguinte comando para rodar a exportação do modelo:

```bash
python export_model.py --model_name gpt2 --learning_rate 5e-5 --apply_lora --quantization --filename model_name --export_dir /path/to/export
```

## Exemplo de Saída:
```css
Modelo TFLite gerado em model_name.tflite
LoRA foi aplicado ao modelo durante a exportação.
```

## 1. Copiar os arquivos necessários

Para garantir que o modelo e os arquivos de tokenização sejam carregados corretamente, copie os seguintes arquivos para os diretórios especificados:

- **Copiar os arquivos**:
  - `merges.txt` e `vocab.json` para o diretório:  
    `llm-on-device/android/app/src/main/assets/gpt2`
  - `model.tflite` e `lora_model.tflite` para o diretório:  
    `llm-on-device/android/app/src/main/assets`

## 2. Construção do projeto

Após copiar os arquivos necessários, você precisará construir o projeto utilizando o **Gradle Wrapper**.

- Abra um terminal e navegue até o diretório raiz do seu projeto.
- Execute o comando a seguir para limpar e construir o projeto:

```bash
  ./gradlew clean build
```

Esse comando limpará qualquer build anterior e criará o APK ou AAR necessário para o seu aplicativo Android.

## 3. Execução do projeto

Após a construção do projeto ser concluída com sucesso, você pode instalar e executar o projeto no seu dispositivo Android ou emulador utilizando o seguinte comando:

- Para instalar o APK no dispositivo ou emulador:

```bash
./gradlew installDebug
```

Com esses passos, o projeto estará preparado e pronto para ser executado no seu dispositivo Android. Certifique-se de que todos os arquivos necessários estejam no local correto antes de iniciar a construção do projeto. 

---

# Notas

Este código foi inspirado em vários tutoriais do TensorFlow, especialmente aqueles focados na utilização de modelos de linguagem pré-treinados e adaptação de modelos para tarefas específicas. A partir dessas fontes, fizemos algumas modificações importantes para adaptar o código às nossas necessidades.

A principal mudança foi a transição de classificação de texto (text classification) para geração de texto (text generation). Isso envolveu adicionar suporte para outro modelo para permitir que ele gerasse sequências de texto, em vez de apenas classificar entradas.

Além disso, foram adicionados novos componentes e funcionalidades para expandir a capacidade do modelo, incluindo:

Método de treinamento: Implementamos um novo método de treinamento, adaptado para suportar o modelo GPT-2 e garantir que ele seja treinado de forma eficiente, especialmente com a adição de LoRA (Low-Rank Adaptation) para otimização de camadas de atenção.

Coleta de métricas de memória: Agora o código coleta métricas relacionadas ao uso de memória, o que ajuda a monitorar o consumo de recursos durante o treinamento e a inferência.

Adição de um tokenizer: Para garantir que o modelo seja capaz de lidar com a tokenização de texto corretamente, foi integrado um tokenizer do GPT-2. Este componente é essencial para transformar o texto em tokens que podem ser processados pelo modelo e para garantir que o processo de inferência seja realizado de forma eficiente.

Essas alterações visam adaptar o modelo de linguagem GPT-2 para o nosso caso específico, otimizando-o tanto para tarefas de geração de texto quanto para execução eficiente em dispositivos móveis com suporte ao formato TFLite.



# Material de Estudo Utilizado

Para a criação deste repositório, foram utilizados os seguintes materiais de estudo e tutoriais, que abordam desde a implementação de **LLMs** (Large Language Models) no Android até a conversão de modelos para **TensorFlow Lite**, passando pela aplicação de **LoRA** (Low-Rank Adaptation) para otimização de inferência em dispositivos móveis.

## 1. Curso da Google sobre LLMs em Android

- [LLM on Android with Keras and TensorFlow Lite | Google for Developers](https://developers.google.com/learn/pathways/llm-on-android)

## 2. Exemplos de Pytorch Models em TensorFlow Lite

- [ai-edge-torch/ai_edge_torch/generative at main · google-ai-edge/ai-edge-torch](https://github.com/google-ai-edge/ai-edge-torch/tree/main/ai_edge_torch/generative/)

## 3. Tutorial Hugging Face para TFLite (com exemplos de QA e Geração de Texto)

- [Hugging Face to TFlite (com exemplos de QA e Text Generation)](https://github.com/huggingface/tflite-android-transformers)


## 4. Inferência de LLM com LoRA

- [LLM Inference with LoRA](https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference)

## 5. LLM Pytorch para TensorFlow Lite em Dispositivos Móveis

- [LLM Pytorch to TensorFlow Lite on Mobile Device](https://developers.googleblog.com/en/ai-edge-torch-generative-api-for-custom-llms-on-device/)

## 6. Tutorial sobre MediaPipe da Google

- [Tutorial sobre MediaPipe da Google](https://github.com/google-ai-edge/mediapipe-samples/tree/main/examples/llm_inference/android)

## 7. Encode e Decoder no Android (Compatível com TensorFlow)

- [Tokenization GPT-2 TF - Hugging Face](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/tokenization_gpt2_tf.py)
- [GPT-2 Tokenizer - Keras](https://keras.io/keras_hub/api/models/gpt2/gpt2_tokenizer/)
- [GPT-2 Tokenizer para Android](https://github.com/huggingface/tflite-android-transformers/tree/master/gpt2/src/main/java/co/huggingface/android_transformers/gpt2/tokenization)


## 8. LiteRT Documentation

- [LiteRT Documentation](https://ai.google.dev/edge/litert)

## 9. Exemplos de Treinamento no Dispositivo

- [Training Example - LiteRT](https://ai.google.dev/edge/litert/models/ondevice_training)

## 10. Quantização de Modelos

- [Quantization - LiteRT](https://ai.google.dev/edge/litert/models/model_optimization)

## 11. Metadados para Modelos de Geração

- [Metadata to Generation Models - LiteRT](https://ai.google.dev/edge/litert/models/metadata)
- [Codegen para Metadados - LiteRT](https://ai.google.dev/edge/litert/android/metadata/codegen)

---

Esses materiais foram fundamentais para a adaptação e exportação de modelos de linguagem para o formato **TensorFlow Lite**, aplicando técnicas como **LoRA**, **quantização** e otimizações específicas para dispositivos móveis.
