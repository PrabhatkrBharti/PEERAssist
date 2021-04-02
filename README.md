## PRAssist
Experimental Code of research paper PRAssist (under review)

# How to run it ... ?

_We recommend using google colab to run it._

_Step 1_ :  Download this repository & Go to PRAssist-master folder
> %cd PRAssist-master

_Step 2_ :  Make bash files executable
> chmod +x requirements.sh
> chmod +x ./raw/extractor.sh
> chmod +x preprocessor.sh
> chmod +x encoder.sh

Step 3 :  Run requirements.sh
> ./requirements.sh

Step 4 :  Run preprocessor.sh
> ./preprocessor.sh

Step 5 :  Run encoder.sh with year (2017 / 2018 / 2019 / 2020) in command line argument , e.g.
> ./encoder.sh 2017

Step 6 :  Now models are ready to go, run any of the two models , paper_rev_sentiment variant
> python model_PRS.py
or , paper_rev variant
> > python model_PR.py


