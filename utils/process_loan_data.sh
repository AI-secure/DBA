# !/bin/bash
# wget https://storage.googleapis.com/kaggle-datasets/34/334209/lending-club-loan-data.zip?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1567788415&Signature=H4eMYgNA6SRYR%2BI7oJnOCHsY9OA2RlOAloWaH0yG%2FumvnyywwPdDVsDrDxQKJoAxWzUF1oNrV6DPJfSvS5d0T74DfHe4zuqJ%2F%2Fec0Wzs1xEEFjdGzzw5FGfuC%2B24lOI5Ql5YuwuRPhAEKUGXaEivCQIZc5%2FoV2dTGWGtbAomVm5og02fW94mNJkTH%2BkY5qhD8PsHOtv7If3aqDOJ2BmkNq9h3STsyUAvWVQHd6HZ00M0PO4eS7WZoPK0mi80kLgLsybMfh9O1H2sBzUmCK1%2Fvy5%2FZgOv%2BKgsLR6usdl9PRP1GTDZVOP7X92nvNuSP7eblRXGBPFYBb5QTeuFbYyXDg%3D%3D -O lending-club-loan-data.zip

unzip lending-club-loan-data.zip -d ./lending-club-loan-data
if [ ! -d ../data  ];then
  mkdir ../data
else
  echo '../data' dir exist
fi

mv ./lending-club-loan-data ../data/
echo move 'lending-club-loan-data' dir to '../data/lending-club-loan-data'
chmod a+r ../data/lending-club-loan-data/loan.csv
python loan_preprocess.py
