# SpeechRepresentations

Scripts to work with the preprocessed intracranial speech representations data that can be downloaded from [here](https://osf.io/qzwsv/), which is used in [this](https://www.biorxiv.org/content/10.1101/2024.08.15.608082v1.full) research article.

## Dependencies
The scripts require Python >= 3.6 and the following packages
* [numpy](http://www.numpy.org/)
* [scipy](https://www.scipy.org/scipylib/index.html)
* [scikit-learn](https://scikit-learn.org/stable/)
* [pandas](https://pandas.pydata.org/) 

## Repository content
To recreate the results from the article, run the following scripts.
* __articulatory.py__: Loads the articulatory data, predicts articulators based on the neural signal, calculates correlations between the original and reconstructed articulators, runs permutations and calculates significant channels.

* __acoustic.py__: Loads the acoustic data, predicts spectrogram based on the neural signal, calculates correlations between the original and reconstructed spectrogram, runs permutations and calculates significant channels.

* __semantic.py__: Loads the semantic data, predicts the neural signal based on the word embeddings, calculates correlations between the original and reconstructed neural signals, runs permutations and calculates significant channels.
