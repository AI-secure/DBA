 
## prepare the dataset:
### LOAN dataset:
in dir `./utils`  

- download the raw dataset `lending-club-loan-data.zip` from `https://www.kaggle.com/wendykan/lending-club-loan-data/` 
- run the script `./process_loan_data.sh` to preprocess the dataset. 

### Tiny-imagenet dataset:
in dir `./utils` 

- download the dataset `tiny-imagenet-200.zip` from `https://tiny-imagenet.herokuapp.com/`
- run `./process_tiny_data.sh` to reformat the dataset.

### others:
MNIST and CIFAR will be automatically download

## run experiments: 

- prepare the pretrained model:
Because we begin to attack after the accuracy in the global model converging, so our pretrained clean models can be downloaded from `https://drive.google.com/file/d/1wcJ_DkviuOLkmr-FgIVSFwnZwyGU8SjH/view?usp=sharing`
(you can also train from the round 0 to obtain the clean model)

- start visdom:
`python -m visdom.server -p 8098`

- run experiments for the four datasets:

`python main.py --params utils/X.yaml`


X = mnist_params, cifar_params, tiny_params or loan_params
parameters can be changed in those yaml files to reproduce our experiments.
