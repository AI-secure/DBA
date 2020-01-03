unzip tiny-imagenet-200.zip -d ./
if [ ! -d ../data  ];then
  mkdir ../data
else
  echo '../data' dir exist
fi

mv ./tiny-imagenet-200 ../data/
echo move 'tiny-imagenet-200' dir to '../data/tiny-imagenet-200'
python tinyimagenet_reformat.py
