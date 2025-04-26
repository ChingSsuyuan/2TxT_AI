# CLIP-Pro: Image Captioning System

A complete framework for training and deploying CLIP-based image captioning models, leveraging CLIP image embeddings and GPT-2 for caption generation.

## Overview

CLIP-Pro is an image captioning system that combines OpenAI's CLIP model for image encoding with GPT-2 for text generation. The system works by:

1. Encoding images using CLIP to get image embeddings
2. Mapping these embeddings to a space compatible with GPT-2's text encoder
3. Using the mapped embeddings as a prefix for GPT-2 to generate relevant captions

This repository includes code for data preparation, model training, and inference, along with a simple web application for testing.

## Components

### 1. Data Preparation (`01_Database_to_Clip.py`)

This script handles downloading images from the COCO dataset and encoding them using the CLIP model:

- Downloads images from COCO dataset
- Extracts captions from annotation files
- Encodes images using CLIP
- Saves embeddings and captions to pickle files

```bash
python 01_Database_to_Clip.py --dataset train --batch-size 2000 --max-images 200
```

### 2. Model Training (`02_Train_Plus.py`) 

This script trains the caption generation model:

- Loads image embeddings and captions
- Trains a mapping network (either MLP or Transformer) to convert CLIP embeddings to GPT-2 compatible embeddings
- Supports training only the mapping network ("prefix") or the entire model (mapping + GPT-2)
- Includes features like gradient clipping, feature noise, and validation monitoring

```bash
python 02_Train_Plus.py --data ./CLIP_Pro_train_merged.pkl --val_data ./CLIP_Pro_val_merged.pkl --mapping_type transformer --prefix_length 40 --epochs 8 --bs 40
```

### 3. Caption Generation (`03_predict.py`)

This script generates captions for new images:

- Loads a trained model
- Processes input images using CLIP
- Generates captions using either beam search or sampling
- Saves results to a JSON file

```bash
python 03_predict.py --img_dir ./test_images --weights ./checkpoints/clip_pro_prefix-best.pt --clip_model RN50x4
```

### 4. Web Application (`app.py`)

A simple Flask web application for uploading images and generating captions:

- Accepts image uploads
- Runs the caption generation script
- Displays results
- Supports translation of captions to different languages

```bash
python app.py
```

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/clip-pro.git
cd clip-pro
```

2. Install the required dependencies:
```bash
pip install torch torchvision clip transformers flask tqdm pillow requests numpy skimage
```

3. Download the CLIP model:
```bash
# The CLIP model will be downloaded automatically when running the scripts
```

4. Prepare the directory structure:
```bash
mkdir -p coco_data/annotations
mkdir -p coco_data/images/train2017
mkdir -p coco_data/images/val2017
mkdir -p checkpoints
mkdir -p test_images
```

## Dataset

This project uses the COCO dataset for training. You'll need to download the annotation files:

```bash
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip -d coco_data/
```

The image files will be downloaded automatically by the data preparation script.

## Training Pipeline

1. **Download and encode images**:
```bash
python 01_Database_to_Clip.py --dataset train --max-images 10000
python 01_Database_to_Clip.py --dataset val --max-images 1000
```

2. **Train the model**:
```bash
python 02_Train_Plus.py --data ./CLIP_Pro_train_merged.pkl --val_data ./CLIP_Pro_val_merged.pkl --mapping_type transformer --prefix_length 40 --epochs 8
```

3. **Generate captions for test images**:
```bash
python 03_predict.py --img_dir ./test_images --weights ./checkpoints/clip_pro_prefix-best.pt
```

## Model Configuration

The model can be configured with various parameters:

- `prefix_length`: Length of the prefix (number of tokens) generated from the image embedding
- `mapping_type`: Type of mapping network (`mlp` or `transformer`)
- `num_layers`: Number of layers in the transformer mapper (if using transformer mapping)
- `normalize_prefix`: Whether to normalize the prefix embeddings
- `feature_noise_scale`: Scale of noise to add to features during training (for regularization)

## Web Application

The included web application provides a simple interface for testing the model:

1. Start the server:
```bash
python app.py
```

2. Access the application at `http://localhost:5002`

3. Upload an image and get a generated caption

4. Optionally translate the caption to different languages

## References

- [CLIP](https://github.com/openai/CLIP): Contrastive Language-Image Pre-Training
- [GPT-2](https://huggingface.co/gpt2): Transformer-based language model
- [COCO Dataset](https://cocodataset.org/): Common Objects in Context dataset

## License

This project is released under the MIT License.