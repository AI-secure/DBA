 
## prepare the dataset:
### loan dataset:
download the raw dataset `lending-club-loan-data.zip` from `https://www.kaggle.com/wendykan/lending-club-loan-data/` into the dir `./utils`   

then run   `cd ./utils`  `./process_loan_data.sh` 

finally there are 51 csv files in`./data/loan/`

### tiny-imagenet dataset:
download the dataset `tiny-imagenet-200.zip` from `https://tiny-imagenet.herokuapp.com/` into the dir `./utils`  
then run `cd ./utils` `./process_tiny_data.sh` to reformat the dataset.

### others:
mnist and cifar will be automaticly download into the dir 
`./data`

## run experiments: 

- download the pretrained model:
Because we begin to attack after the accuracy in the global model converging, so our pretrained clean models can be downloaded from `https://drive.google.com/file/d/1wcJ_DkviuOLkmr-FgIVSFwnZwyGU8SjH/view?usp=sharing`
(you can also train from the round 0 to obtain the clean model)

- start visdom:
`python -m visdom.server -p 8098`

- run experiments for the four datasets:

`python main.py --params utils/mnist_params.yaml`

`python main.py --params utils/cifar_params.yaml`

`python main.py --params utils/tiny_params.yaml`

`python main.py --params utils/loan_params.yaml`

parameters can be changed according to the comments in those yaml files to reproduce our experiments.
