# Pretraining of BEiT

The following steps has been extracted from [this](https://github.com/microsoft/unilm/tree/master/beit) repository.

## Steps

The following steps should be made to be able to run the pretraining successfully

### Model OUTPUT_DIR

We have to create the model output directory and add it as an env-variable, where the pretrained-models will be stored. First create the output directory:

> mkdir -p <model_output_dir>

Then add the directory as an environment variable

> OUTPUT_DIR=<model_output_dir>

### Data DATA_PATH

Then we have to create the env-variable and folder and add the data that we want to be used in the pre-training.

> DATA_PATH=<data_directory>
> mkdir -p ${DATA_PATH}

Then get the training data from the generated tiles folder. We are only using the .npy files from this folder. We then convert them into a .jpeg image, because the pre-training process uses PIL to load the images. The process of converting and adding the images to the data folder can be done using the numpy_to_image.py file.

In order to generate the images, use the existing 'numpy_to_image.py' script. The script takes in two arguments, the source folder, where the numpy files exists.
> python numpy_to_image.py <source_data_path> ${DATA_PATH}

### DALL-E TOKENIZER_PATH

The pre-training process uses the DALL-E image tokenizer, which is why we need to load the encoder and decoder pickle from openai. First we need to define the path where we can find the tokenizer pickles.

> TOKENIZER_PATH=<tokenizer_path>

> mkdir -p ${TOKENIZER_PATH}

Then we need to get the pickles.

> wget -O ${TOKENIZER_PATH}/encoder.pkl https://cdn.openai.com/dall-e/encoder.pkl && wget -O ${TOKENIZER_PATH}/decoder.pkl https://cdn.openai.com/dall-e/decoder.pkl

## Running the pre-training

In order to run the pre-training script, we need some arguments to make it work. These arguments are adjustable, and below I've provided an example of a default script.

> run_beit_pretraining.py \
        --data_path ${DATA_PATH} --output_dir ${OUTPUT_DIR} --num_mask_patches 75 \
        --model beit_base_patch16_224_8k_vocab --discrete_vae_weight_path ${TOKENIZER_PATH} \
        --batch_size 16 --lr 1.5e-3 --warmup_steps 10 --epochs 30 \
        --clip_grad 3.0 --drop_path 0.1 --layer_scale_init_value 0.1 --device cpu


* __--num_mask_patches__:               Number of the input patches that needs to be masked
* __--batch_size__:                     Batch size per GPU
* __--lr__:                             Number of the input patches that needs to be masked
* __--warmup_steps__:                   Learning rate warmup steps
* __--epochs__:                         Total pre-training epochs
* __--clip_grad__:                      Clip gradient norm
* __--drop_path__:                      Stochastic depth rate
* __--imagenet_default_mean_and_std__:  Enable this for ImageNet-1k pre-training, i.e., (0.485, 0.456, 0.406) for mean and (0.229, 0.224, 0.225) for std. We use (0.5, 0.5, 0.5) for mean and (0.5, 0.5, 0.5) for std by default on other pre-training data.
* __--layer_scale_init_value__:         0.1 for base, 1e-5 for large, set 0 to disable layerscale
