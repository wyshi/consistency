## 0. Install the environment
```conda env create -f environment.yml```

```python -m spacy download en_core_web_sm```


## download the imitation classifier and dialog act classifier
download [here](https://drive.google.com/file/d/1kLuLme1fS8hTphf-ebgfPCRKHovSX_zc/view?usp=sharing)

[labelencoder_A](https://drive.google.com/file/d/1tb2MnbZVx7gbWgStxUNQLJvEjyHbb8l7/view?usp=sharing)

[labelencoder_B](https://drive.google.com/file/d/1lnyGEOAgWVHH3-NYl7S3ppcOJExVIVpZ/view?usp=sharing)

```
python imitation_learning/load_model.py
```

## RL, PPO
```
python PPO.py
```

## Persuasion agent class
PersuasionInteract.py: a persuasion chatbot, also the actor in PPO.py

## a good reference
https://github.com/qywu/TextGAIL


## future steps
Automate the inconsistency/repetition detection steps, other related tasks