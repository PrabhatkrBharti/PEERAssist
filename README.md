# PRAssist
Experimental Code of research paper PRAssist (under review)

# How to run it ... ?

_We recommend using google colab to run it._
***

_Step 1_ :  **Download master-branch** of this repository & Go to **main folder** of it.

_Step 2_ :  **Make bash files executable**
```bash
chmod +x requirements.sh
chmod +x ./raw/extractor.sh
chmod +x preprocessor.sh
chmod +x encoder.sh
```
_Step 3_ :  **Run requirements.sh**
```basg
./requirements.sh
```
_Step 4_ :  **Run preprocessor.sh**
```bash
./preprocessor.sh
```
_Step 5_ :  **Run encoder.sh** with year (2017 / 2018 / 2019 / 2020) in command line argument , e.g.
```bash
./encoder.sh 2017
```
_Step 6_ :  Now models are ready to go, **run any of the two models with year**, paper_rev_sentiment variant
```bash
python model_PRS.py 2017
```
or , paper_rev variant
```bash
python model_PR.py 2017
```

