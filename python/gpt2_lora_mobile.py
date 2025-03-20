import tensorflow as tf
from transformers import (
    TFGPT2LMHeadModel,
    GPT2Config,
    GPT2Tokenizer,
)
import argparse

LORA_RANK = 64
def get_argsparse():
    """
        args (argparse.Namespace): Parâmetros de entrada fornecidos pela linha de comando, incluindo:
        - model_name (str): Nome do modelo pré-treinado (ex: "gpt2").
        - learning_rate (float): Taxa de aprendizado para o otimizador.
        - apply_lora (bool): Se True, aplica LoRA (Low-Rank Adaptation) nas camadas de atenção do modelo.
        - quantization (bool): Se True, aplica a quantização do modelo para reduzir seu tamanho e melhorar a eficiência em dispositivos móveis.
        - filename (str): Nome base do arquivo de saída.
        - export_dir (str): Diretório para exportação do modelo salvo.
    """
    argparser = argparse.ArgumentParser("Configuration")
    argparser.add_argument("--quantization", '-q', action='store_true')
    argparser.add_argument("--seqlen", '-sl', default=32, type=int, help="Max sequence length")
    argparser.add_argument("--learning_rate", '-lr', default=1e-5, type=float, help="Learning rate")
    argparser.add_argument("--model_name", "-mn", default='openai-community/gpt2', type=str, help="Model on huggingface. If changes, you should adapt the code")
    argparser.add_argument("--export_dir", "-ed", default="saved_model", type=str, help="Export directory to save model info")
    argparser.add_argument("--filename", '-f', default='model.tflite', type=str, help="File name to save model")
    argparser.add_argument("--apply_lora", '-lora', action="store_true")

    return argparser.parse_args()
class LoraAttn(tf.keras.layers.Layer):
    """
        Classe que implementa a técnica de LoRA (Low-Rank Adaptation) para camadas de atenção no TensorFlow.

        Esta implementação aplica LoRA na camada de atenção do modelo, adaptando sua função de atenção através da introdução de uma perturbação de baixo-rank, representada pelas matrizes `A` e `B`. Essas matrizes são aprendidas durante o treinamento e controlam a adaptação da camada de atenção original.

        A classe é projetada para ser utilizada diretamente dentro de uma arquitetura de rede neural baseada em Transformer, como o GPT-2, onde o número de heads de atenção e a dimensionalidade das camadas são bem conhecidos.

        A classe realiza as seguintes operações:
        1. Inicializa as matrizes `A` e `B` que formam a adaptação de baixo-rank.
        2. Aplica uma modulação da camada de atenção original usando essas matrizes.
        3. A integração das modificações de LoRA no gráfico computacional do TensorFlow é garantida com a adição de perdas nulas, permitindo que o gradiente seja calculado corretamente durante o treinamento.
    """
    def __init__(self, attn_layer: tf.keras.layers.Layer, rank: int, layer_id:int):
        super(LoraAttn, self).__init__(name="lora_layer")
        self.old_layer = attn_layer
        self.old_weights = self.old_layer.get_weights()
        self.rank = rank

        self.A = self.add_weight(
            shape=(768, self.rank), # 768 refere-se à dimensão da entrada de cada cabeça de atenção (no caso do GPT-2)
            initializer='glorot_uniform',
            trainable=True,
            name=f"lora_qa_{layer_id}"
        )

        self.B = self.add_weight(
            shape=(self.rank, 2304), # 2304 refere-se à dimensão da saída da camada de atenção
            initializer='zeros',
            trainable=True,
            name=f"lora_qb_{layer_id}"
        )

        self.lora_alpha = 1.0
        
    def call(self, inputs):
        """
            Aplica a modulação LoRA na camada de atenção original.
            A camada de atenção original é computada normalmente e depois ajustada com as modificações de baixo-rank introduzidas pelas matrizes `A` e `B`.
            inputs: O tensor de entrada para a camada de atenção.
            Retorna o Tensor modificado que combina a saída da atenção original com a adaptação de baixo-rank.
        """
        # Passo 1: Obtém a saída da camada de atenção original
        original = self.old_layer(inputs)
        # Passo 2: Aplica a modulação de baixo-rank LoRA
        output = tf.matmul(tf.matmul(inputs, self.A), self.B) * self.lora_alpha
        
        # Passo 3: Força a inclusão das novas camadas LoRA no grafo computacional do TensorFlow
        # Isso é necessário para garantir que o TensorFlow reconheça as operações e calcule gradientes corretamente.
        self.add_loss(0.0 * tf.reduce_sum(self.A))
        self.add_loss(0.0 * tf.reduce_sum(self.B))
        
        # Passo 4: Retorna a combinação da saída original com a adaptação LoRA
        return original + output


class GPT2Generator(tf.Module):
    """
        Classe para geração de texto baseada no modelo GPT-2, com suporte à adaptação LoRA para ajustar camadas de atenção do modelo sem a necessidade de re-treinamento completo.

        Esta classe integra o modelo GPT-2 pré-treinado, e oferece suporte para treinamento e inferência. A adaptação LoRA é aplicada nas camadas de atenção, permitindo um ajuste eficiente do modelo para novos dados ou tarefas específicas com um custo computacional reduzido.

        A implementação é otimizada para compatibilidade com LiteRT, permitindo a execução eficiente em dispositivos móveis e incorporados.

        Métodos principais:
        - **apply_lora()**: Aplica a adaptação LoRA nas camadas de atenção do modelo GPT-2, congelando os pesos das camadas originais e permitindo que apenas os pesos das camadas LoRA sejam treinados.
        - **infer()**: Realiza a inferência, ou seja, gera previsões de tokens do modelo com base em entradas fornecidas.
        - **train()**: Realiza o treinamento do modelo, aplicando os gradientes apenas aos pesos LoRA treináveis e utilizando o otimizador AdamW.
        - **get_w()**: Retorna os pesos do modelo, útil para a análise e monitoramento dos parâmetros do modelo durante o treinamento.

        A classe permite personalização da aplicação LoRA nas camadas de atenção e configuração da taxa de aprendizado.
    """
    def __init__(self, model_name = None, lr=None, apply_lora=False):
        super().__init__()
        self.model = TFGPT2LMHeadModel.from_pretrained(model_name)
        self.optimizer = tf.keras.optimizers.AdamW()
        if apply_lora:
            self.apply_lora()

    def apply_lora(self):
        """
            Aplica a adaptação LoRA nas camadas de atenção do modelo GPT-2. Essa técnica ajusta as camadas de atenção de baixo-rank para otimizar a adaptação do modelo a novas tarefas com um número reduzido de parâmetros treináveis.

            **Notas Importantes**:
            - A arquitetura do modelo precisa ser conhecida de antemão, pois a adaptação LoRA é aplicada manualmente nas camadas.
            - As camadas originais de atenção são congeladas, e apenas as camadas LoRA são treináveis.
            - Durante a adaptação LoRA, o modelo é recompilado para garantir que as novas camadas LoRA sejam corretamente registradas no gráfico computacional do TensorFlow.

            **Observação**:
            - O congelamento das camadas é feito manualmente, utilizando o atributo `_trainable`, para garantir que as camadas LoRA possam ser treinadas separadamente.
        """
        for i, tfblock in enumerate(self.model.transformer.h):
            original_attn = tfblock.attn.c_attn  # Acessa a camada original de atenção
            original_attn.trainable = False  # Congela a camada original de atenção
            
            # Congela as variáveis da camada original e remove o gradiente
            for var in original_attn.variables:
                var.assign(tf.stop_gradient(var))  # Remove o gradiente
                var._trainable = False  # Congela os pesos da camada original

            # Cria a camada LoRA e a integra à camada original de atenção
            lora_layer = LoraAttn(original_attn, rank=8, layer_id=i)
            for var in lora_layer.variables:
                # Apenas os pesos LoRA são treináveis
                if "lora" in var.name:
                    var._trainable = True
                else:
                    var._trainable = False  # Congela qualquer outro peso herdado

            # Substitui a camada original pela camada LoRA adaptada
            setattr(tfblock.attn, "c_attn", lora_layer)


        # Congela os embeddings e a última camada LayerNorm
        self.model.transformer.wte.trainable = False  # Embeddings de tokens
        self.model.transformer.wpe.trainable = False  # Embeddings de posições
        self.model.transformer.ln_f.trainable = False  # Última LayerNorm

        # Congela as camadas de normalização e feedforward do modelo
        for i, tfblock in enumerate(self.model.transformer.h):
            tfblock.ln_1.trainable = False  # LayerNorm 1
            tfblock.ln_2.trainable = False  # LayerNorm 2
            tfblock.attn.c_proj.trainable = False  # Projeção da atenção
            tfblock.mlp.c_fc.trainable = False  # Feedforward do MLP
            tfblock.mlp.c_proj.trainable = False  # Projeção final do MLP

        # Recompila o modelo para registrar as camadas LoRA no gráfico computacional
        optimizer = self.model.optimizer if self.model.optimizer else tf.keras.optimizers.Adam()
        self.model.compile(optimizer=optimizer)
    
    @tf.function(
        input_signature=[
            tf.TensorSpec([None, None], tf.int32),
            tf.TensorSpec([None, None], tf.int32)
        ]
    )
    def infer(self, input_ids, attention_mask):
        """
            Realiza a inferência do modelo para gerar a próxima previsão de tokens.

            Args:
                input_ids (tf.Tensor): IDs de entrada dos tokens.
                attention_mask (tf.Tensor): Máscara de atenção para a sequência de entrada.

            Returns:
                dict: Contém os logits da previsão de tokens.
        """
        # Realiza a previsão, retornando a última previsão de token
        tokens = self.model(input_ids=input_ids, attention_mask=attention_mask, training=False).logits[0:, -1]
        return {
            "logits": tokens
        }

    @tf.function(
        input_signature=[
            tf.TensorSpec([None, None], tf.int32),
            tf.TensorSpec([None, None], tf.int32),
            tf.TensorSpec([None, None], tf.int32),
        ]
    )
    def train(self, input_ids, attention_mask, labels):
        """
            Função de treinamento para aplicar gradientes aos pesos LoRA treináveis.
            A função realiza um ciclo de feedforward e backward apenas nos pesos LoRA treináveis.
            Args:
                input_ids (tf.Tensor): IDs dos tokens de entrada.
                attention_mask (tf.Tensor): Máscara de atenção.
                labels (tf.Tensor): Labels para treinamento supervisionado.
            Returns:
                dict: Contém a perda calculada durante o treinamento.
        """
        with tf.GradientTape() as tape:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                training=True
            )
            loss = outputs.loss
        # Aplica os gradientes aos pesos LoRA treináveis
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return {"loss": loss}

    @tf.function
    def get_w(self):
        weights = {}
        for var in self.model.weights:
            weights[var.name] = var
        return weights

    
def export_saved_model(args):
    """
        Função para exportar um modelo treinado (ou adaptado) para o formato TFLite, com suporte a LoRA e quantização opcional.

        Essa função realiza o seguinte:
        1. Cria um modelo `GPT2Generator` com base nos parâmetros fornecidos (incluindo LoRA e quantização).
        2. Define as assinaturas das funções de treinamento, inferência e obtenção de pesos para a exportação do modelo.
        3. Salva o modelo no formato `SavedModel` com as assinaturas apropriadas.
        4. Converte o modelo para o formato TFLite, aplicando quantização, se solicitado.
        5. Salva o modelo TFLite gerado em um arquivo binário.
    """
    file_name = f'lora_{args.filename}' if args.apply_lora else args.filename
    file_name = f'q_{args.filename}' if args.quantization else file_name

    model = GPT2Generator(model_name=args.model_name, lr=args.learning_rate, apply_lora=args.apply_lora)
    # Define as assinaturas das funções de treinamento, inferência e obtenção de pesos
    signatures = {
        "train":  model.train.get_concrete_function(),
        "infer":  model.infer.get_concrete_function(),
        "get_weights": model.get_w.get_concrete_function(),
    }
    # Salva o modelo no formato SavedModel com as assinaturas
    tf.saved_model.save(
        obj=model,
        export_dir=args.export_dir,
        signatures=signatures
    )
    # Salva o tokenizer associado ao modelo GPT-2
    # Isso vai ser importante quando formos passar o modelo para o dispositivo, uma vez que replicamos o tokenizer no kotlin
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.save_pretrained(f"tokenizer_{args.filename}")

    converter = tf.lite.TFLiteConverter.from_saved_model(args.export_dir)

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.experimental_enable_resource_variables = True
    if args.quantization:
        converter.optimizations = [tf.lite.Optimize.DEFAULT] ## Aparetemente operadores com input e output do tipo flooat, não são suportados por INT8, causando erro.
        converter.target_spec.supported_types = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        # converter.inference_input_type = tf.uint8
        # converter.inference_output_type = tf.uint8

    tflite_model = converter.convert()
    with open(file_name, "wb") as f:
        f.write(tflite_model)

    print("Modelo TFLite gerado em model_distilgpt2.tflite")
    if args.apply_lora:
        print("")


if __name__ == "__main__":
    args = get_argsparse()
    export_saved_model(args)
