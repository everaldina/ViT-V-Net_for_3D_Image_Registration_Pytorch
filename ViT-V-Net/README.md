# Como rodar o ViT-V-Net

## Sumário
1. [Preparando o ambiente](#1-preparando-o-ambiente)
2. [Treino](#2-treino)
    - [Paramentros de treino](#paramentros-de-treino)
    - [Continuação de treino](#continuação-de-treino)
3. [Alinhamento de imagens](#3-alinhamento-de-imagens)
    - [Parametros de alinhamento](#parametros-de-alinhamento)
    - [Exemplo de execução](#exemplo-de-execução)
    - [Saida](#saida)
4. [Aplicar fluxo de Alinhamento](#4-aplicar-fluxo-de-alinhamento)
    - [Paramentros de aplicação de alinhamento](#paramentros-de-aplicação-de-alinhamento)
    - [Exemplo de execução](#exemplo-de-execução)
    - [Saida](#saida-1)


## 1. Preparando o ambiente
Para preparar o ambiente de execucão é preciso criar uma imagem docker com todas as dependências necessárias. Para isso, execute o seguinte comando:
1. Crie a imagem docker
    ```bash
    cd container/vit-v-net
    docker build -t vit-v-net .
    ```

2. Execute o container
- O arquivo `container/vit-v-net/how-to-run.txt` contem o comando de execução do container na estação de trabalho. 
- Substituir o `/home/Diagnostico_Segmentacao-Source/ViT-V-Net_for_3D_Image_Registration_Pytorch/ViT-V-Net` pelo caminho do diretório do `ViT-V-Net_for_3D_Image_Registration_Pytorch\ViT-V-Net` na estação de trabalho onde o container será executado.
- Substituir o `/home/data/orCaScore/vit-v-net/T26_normal` pelo caminho do diretório onde estão os dados de treino e teste.

## 2. Treino
1. Acesse o workspace do container
    ```bash
    cd /workspace/ViT-V-Net
    ```
2. Execute o script de treino, passando os parâmetros necessários, abaixo um exemplo de execução:
    ```bash
    python train.py --train_dir /workspace/data/train/ --save_dir T26_normal --log_name T26_normal_log
    ```

### Paramentros de treino
- `--train_dir`: Diretório onde estão os dados de treino.
- `--save_dir`: Diretório onde será salvo o modelo treinado.
- `--log_name`: Nome do arquivo de log.
- `--batch_size`: Tamanho do batch.
- `--lr`: Taxa de aprendizado.
- `--max_epoch`: Número máximo de épocas.
- `--continue_train`: Continuar treino de um modelo salvo.
- `--epoch_start`: Época onde o treino será iniciado.

### Continuação de treino
- O Vit-V-Net não carrega modelos por numero de epocas, os arquivos sao salvos em um padrao `dsc{loss}.pth.tar`, onde `loss` é o valor da loss do modelo salvo, ao carregar o modelo para continuar o treino, o script carrega o modelo com menor loss.
- São slavos os 8 melhores modelos e quando um modelo com menor loss é salvo, o modelo com maior loss é deletado.
- Para continuar um treinamento é necessário passar o parametro `--continue_train` e `--epoch_start` com a epoca onde o treino deve ser continuado.
- Exemplo de continuação de treino:
    ```bash
    python train.py --train_dir /workspace/data/train/ --save_dir T26_normal --log_name T26_normal_log --continue_train --epoch_start 10
    ```

## 3. Alinhamento de imagens
1. Preparação de diretorio de teste
 - Deve ser um diretorio com arquivos de extensoes `.pkl` que contem as imagens a serem alinhadas.
 - O pickle deve ser uma tupla (x, y) onde x é a imagem a ser alinhada e y é a imagem de referencia.
 - O nome de cada arquivo sera o nome do diretorio onde o resultado sera salvo.
2. Preparação do diretorio do modelo
 - Dentro do diretorio `experiments` é necessário ter um diretorio em que o modelo treinado foi salvo.
 - Não é preciso que seja um diretorio filho direto de `experiments`, pode ser um diretorio filho de um diretorio filho de `experiments`.
 - O modelo salvo deve ter extensão `.pth.tar`.
 - É nesse diretorio que o script de alinhamento irá salvar os resultados.
3. Execute o script de alinhamento, passando os parâmetros necessários.


### Parametros de alinhamento
- `--test_dir`: Diretório onde estão os dados de teste. Deve seguir a estrutura de diretórios descrita em [Preparação de diretorio de teste](#preparação-de-diretorio-de-teste).
    - **default**: `/vit-v-net/test/`
- `--model_folder`: Diretório onde está o modelo treinado. Deve seguir a estrutura de diretórios descrita em [Preparação do diretorio do modelo](#preparação-do-diretorio-do-modelo).
    - **default**: `vit-v-net/`
- `--cascade_number`: Número de cascadas para o alinhamento.
    - **default**: `1`
- `--num_workers`: Número de workers para o dataloader.
    - **default**: `1`

### Exemplo de execução

- Pickles da pasta `T26_transformed/test/` serão alinhados com o modelo salvo em `experiments/T26_transformed` com 4 cascadas.
    ```bash
    python align.py --test_dir /workspace/data/orCaScore/vit-v-net/T26_transformed/test/ --model_folder T26_transformed --cascade_number 4
    ```

- Pickles da pasta `T26_normal/test/` serão alinhados com o modelo salvo em `experiments/T26_normal` com 4 cascadas.
    ```bash
    python align.py --test_dir /workspace/data/orCaScore/vit-v-net/T26_normal/test/ --model_folder T26_normal --cascade_number 4
    ```

### Saida
- As saidas do alinhamento estarão no diretorio `experiments/{model_folder}/` e seguem essa estrutura:

    ```
    experiments
    │
    └─── {model_folder}
        │
        ├──── `model.pth.tar`
        ├─── {img_1}
        │   ├── cascade_1
        │   │   ├── x.pkl
        │   │   ├── y.pkl
        │   │   ├── flow.pkl
        │   │   ├── x_to_y.pkl
        │   │   ├── y_to_x.pkl
        │   │   └── cascade_1_results.pkl
        │   ├── cascade_2
        │   │   ├── x.pkl
        │   │   ├── y.pkl
        │   │   ├── flow.pkl
        │   │   ├── x_to_y.pkl
        │   │   ├── y_to_x.pkl
        │   │   └── cascade_2_results.pkl
        │   └── ...
        ├─── {img_2}
        │   ├── cascade_1
        │   │   ├── x.pkl
        │   │   ├── y.pkl
        │   │   ├── flow.pkl
        │   │   ├── x_to_y.pkl
        │   │   ├── y_to_x.pkl
        │   │   └── cascade_1_results.pkl
        │   ├── cascade_2
        │   │   ├── x.pkl
        │   │   ├── y.pkl
        │   │   ├── flow.pkl
        │   │   ├── x_to_y.pkl
        │   │   ├── y_to_x.pkl
        │   │   └── cascade_2_results.pkl
        │   └── ...
        └─── ...

    ```
- `x.pkl` e `y.pkl` são as imagens de entrada da cascada.
- `flow.pkl` é o campo de deslocamento calculado pela para alinha `x` em `y`.
- `x_to_y.pkl` é a imagem `x` alinhada em `y`.
- `y_to_x.pkl` é a imagem `y` alinhada em `x`.
- `cascade_{n}_results.pkl` é o resultado da cascada `n`
    - Uma tupla (x, y) onde x é a imagem alinhada em y e y é a imagem de referencia.
    - É essa imagem que será usada como entrada para a próxima cascada.


## 4. Aplicar fluxo de Alinhamento
1. Preparação do transform_path
    - O caminho deve ser o mesmo do `model_folder` do alinhamento discrito em [Saída](#saida).
    - Arquivos que devem estar presentes:
        - `y.pkl`: Dentro de cada diretório de cascata deve haver um arquivo `y.pkl` que é a imagem de target. Ela serve para redimensionar a imagem de entrada.
        - `flow.pkl`: Dentro de cada diretório de cascata deve haver um arquivo `flow.pkl` que sera aplicado a imagem de entrada.
    - Não é necessario que todos os arquivos de alinhamento estejam presentes, apenas os citados acima e a estrutura de diretórios deve ser a mesma.
2. Execute o script de aplicação de alinhamento, passando os parâmetros necessários.


### Paramentros de aplicação de alinhamento
- `data_folder`: Diretório onde estão os dados a serem alinhados.
    - Deve ter arquivos de extensão `.pkl` ou `.nii` que contem as imagens a serem alinhadas.
    - Cada nome de arquivo deve ter um diretório correspondente no `transform_path`.
    - **Parametro obrigatório.**
- `transform_path`: Diretório onde estão salvos os campos de deslocamento e imagens de referência de cada cascata.
    - **Parametro obrigatório.**
- `--result_path`: Diretório onde serão salvos os resultados.
    - **default**: `aligned/`
- `--reshape_mode`: Modo de redimensionamento das imagens, pode ser `zoom` ou `reduce`.
    - **default**: `zoom`
- `--image_type`: Tipo de imagem a ser alinhada, pode ser `normal` ou `transformed`.
    - No caso de `transformed` a imagem de entrada será redimensionada para o tamanho da imagem de referencia e depois redimensionada para (64, 512, 512).
    - No caso de `normal` a imagem de entrada será redimensionada para o tamanho da imagem de referencia e depois redimensionada para (64, 512, 512).
    - **default**: `normal`
- `--suffix`: Sufixo a ser adicionado ao nome do arquivo de saída.
    - Os arquivos de saida sao salvos na pasta `result_path` com o nome `cascade_{n}.pkl`, ou caso o parametro `suffix` seja passado, `cascade_{n}_{suffix}.pkl`.
    - **default**: `None`
- `--file_extension`: Extensão do arquivo a ser alinhado, pode ser `pkl` ou `nii`.
    - **default**: `nib`
- `--num_workers`: Número de workers para o dataloader.

### Exemplo de execução
- Aplicar alinhamento em imagens de um diretório `square_experiment` com os campos de deslocamento salvos em `experiments/T26_transformed` e salvar os resultados em `square_experiment/results`. Usar redimensionamento `zoom` e extensão de arquivo `pkl`.
    ```bash
    python apply_align.py /workspace/data/orCaScore/vit-v-net/square_experiment /workspace/ViT-V-Net/experiments/T26_transformed --result_path /workspace/data/orCaScore/vit-v-net/square_experiment/results --image_type transformed --reshape_mode zoom --file_extension pkl
    ```

- Aplicar alinhamento em imagens de um diretório `artery_labels` com os campos de deslocamento salvos em `experiments/T26_transformed` e salvar os resultados em `artery_labels/results`. Usar redimensionamento `zoom` e extensão de arquivo `pkl`. Adicionar sufixo `transformed_zoom` ao nome do arquivo de saída.
    ```bash
    python apply_align.py /workspace/data/orCaScore/artery_labels /workspace/ViT-V-Net/experiments/T26_transformed --result_path /workspace/data/orCaScore/artery_labels/results --suffix transformed_zoom --image_type transformed --reshape_mode zoom
    ```

### Saida
- As saidas do alinhamento estarão no diretorio `result_path` e seguem essa estrutura:

    ```
    result_path
    │
    ├─── {img_1}
    │   ├── cascade_1.pkl
    │   ├── cascade_2.pkl
    │   └── ...
    ├─── {img_2}
    │   ├── cascade_1.pkl
    │   ├── cascade_2.pkl
    │   └── ...
    └─── ...

    ```
- Caso o parametro `suffix` seja passado, o nome do arquivo de saida será `cascade_{n}_{suffix}.pkl`.


