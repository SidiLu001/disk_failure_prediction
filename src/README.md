## Usage

### Install requirement libraries

```
pip install -r requirement.txt
```

### Data Sample

Due to the complex procedures of data cleaning and processing, we do not have a complete and systematic code to share for these procedures, but we present a detailed description for the guidance in the paper, and we provide some data samples [here](https://drive.google.com/drive/folders/15UsIj9y4bc0Be-HMyX4BegUyq1HCs5LZ?usp=sharing) for the execution of models. 





### CNN-LSTM

```
python3 CNNLSTM.py --xpath X.npy --ypath y.npy --group SPL
```

### LSTM

```
python3 LSTM.py --xpath X.npy --ypath y.npy --group SPL
```

### RF

```
python3 RandomForest.py --xpath X.npy --ypath y.npy --group SPL
```

### GBDT

```
python3 GBDT.py --xpath X.npy --ypath y.npy --group SPL
```

### BAYES

```
python3 BAYES.py --xpath X.npy --ypath y.npy --group SPL
```

