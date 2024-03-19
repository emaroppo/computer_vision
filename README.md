Stash of simple pytorch implementations of common ml problems. The purposes of this repo are:
1. Getting scripts out there quickly, even without a supporting frontend (input welcome); some may be expanded into full projects
2. Establish code references on how to implement common architecures and/or handle input/output data of popular pretrained models, as well as adapting them to new problems

## Computer Vision
- Face Detection and Recognition: based on https://github.com/timesler/facenet-pytorch/
    - TO DO: create face db (maybe vector db?)
- Plant Disease Classification: simple cnn classifier with the twist of taking the plant species as input alongside the image (trained on PlantVillage)
- TO DO: Image Segmentation
- TO DO: Object Detection

## RAG
- Semantic Search for Movie Database: based on freecodecamp's tutorial (https://www.youtube.com/watch?v=JEBDfGqrAUA), reimplemented to use faiss and a local mongodb instance for a better reason than me not being bothered to set up an atlas instance. A different, more interesting one. (WIP)

## PEFT
- LoRA finetuning script: based on https://www.youtube.com/watch?v=Us5ZFp16PaU&t=211s (WIP)
- QLora finetuning script: based on https://www.youtube.com/watch?v=XpoKB3usmKc (WIP)