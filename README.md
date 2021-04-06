# PRAssist
Experimental Code of research paper PRAssist (under review)

# How to run it ... ?

_We recommend using google colab to run it._
***

_Step 1_ :  **Download master-branch** of this repository [ zip file size : ~ 60 Mb ] & Go to **main folder** of it.

_Step 2_ :  **Make bash files executable**
```bash
chmod +x requirements.sh
chmod +x extractor.sh
chmod +x preprocessor.sh
chmod +x encoder.sh
chmod +x clear_year.sh
```
_Step 3_ :  **Run requirements.sh** to download all requirements.
```bash
./requirements.sh
```
_Step 4_ : **Run extractor.sh** to extract all compressed raw data files.
```bash
./extractor.sh
```
[_optional step_] : **Run get_stats.py** to generate statistics given in paper in table I.
```bash
python get_stats.py
```
_Step 5_ :  **Run preprocessor.sh** ,  takes around [10 - 15 min.] to preprocess all data.
```bash
./preprocessor.sh
```
_Step 6_ :  **Run encoder.sh** with year (2017 / 2018 / 2019 / 2020) in command line argument [~ 30 min.]  to generate embeddings , e.g.
```bash
./encoder.sh 2017
```
_Step 7_ :  Now models are ready to go, **run any of the two models**, run paper_rev_sentiment variant
```bash
python model_PRS.py 2017
```
or , paper_rev variant
```bash
python model_PR.py 2017
```
on ICLR-2017/18/19/20 Dataset.


[_optional step_] :  When **running low on memory** , clear embedding datas of the dataset which is not required further.
```bash
./clear_year.sh 2017
```
***
