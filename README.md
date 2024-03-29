# DiplomaThesis
## Czech NLP with Contwextalized Embeddings

This repository contains code and text for my diploma thesis and also best models for all task variants. Models are published under Attribution-NonCommercial-ShareAlike 4.0 International licence.

Best models are available in the form of checkpoints temporarily on AIC cluster:
* tagging and lemmatization - tl_18 [index](aic.ufal.mff.cuni.cz/~doubrap1/ch18.index) [data](aic.ufal.mff.cuni.cz/~doubrap1/ch18.data-00000-of-00001)
* csfd [index](aic.ufal.mff.cuni.cz/~doubrap1/sentiment_analysis.py-2021-07-05_115521-a=16,bs=2,b=...index) [data](aic.ufal.mff.cuni.cz/~doubrap1/sentiment_analysis.py-2021-07-05_115521-a=16,bs=2,b=...data-00000-of-00001) [mappings](aic.ufal.mff.cuni.cz/~doubrap1/mappings.pickle)
* mall [index](aic.ufal.mff.cuni.cz/~doubrap1/sentiment_analysis.py-2021-06-08_234151-a=12,bs=4,b=...index) [data](aic.ufal.mff.cuni.cz/~doubrap1/sentiment_analysis.py-2021-06-08_234151-a=12,bs=4,b=...data-00000-of-00001)
* facebook [index](aic.ufal.mff.cuni.cz/~doubrap1/sentiment_analysis.py-2021-06-08_172844-a=12,bs=4,b=...index) [data](aic.ufal.mff.cuni.cz/~doubrap1/sentiment_analysis.py-2021-06-08_172844-a=12,bs=4,b=...data-00000-of-00001)
* joint [index](aic.ufal.mff.cuni.cz/~doubrap1/sentiment_analysis.py-2021-07-02_181019-a=32,bs=1,b=...index) [data](aic.ufal.mff.cuni.cz/~doubrap1/sentiment_analysis.py-2021-07-02_181019-a=32,bs=1,b=...data-00000-of-00001)

or on Lindat:
* [Sentiment](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4601)
* [POS Tagging and Lemmatization](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4613)

Demo notebook with an example of usage of pretrained models is available for tagging and lemmatization [here.](https://github.com/flower-go/DiplomaThesis/blob/b02bdb7288090379ffab5bdf624d7453a2a8a5e8/Taggin_and_Lemmatization_Working_Example.ipynb)
Demo for sentiment is available [here](https://github.com/flower-go/DiplomaThesis/blob/8d408e6a122d33b651dd0d37937969cc6c4f350a/sentiment_example.ipynb).

If you wish to replicate training experiments, the list of scripts with hyperparameters is in [run_scripts](https://github.com/flower-go/DiplomaThesis/blob/51cce14ea3d6834c9249325016f9cf4b80af871d/run_experiments)
Input data should be in the following format: every line contains one input word, gold lemma and gold tag (all separated by tab) as in the following example.  
Faxu	fax	NNIS3-----A----  
škodí	škodit_:T	VB-P---3P-AA---  
především	především	Db-------------  
přetížené	přetížený_^(*3it)	AAFP1----1A----  
telefonní	telefonní	AAFP1----1A----  
linky	linka	NNFP1-----A----  

The model also needs the same embeddings as in the demo notebooks. 

