![CI](https://github.com/JonasFreibur/ledebruiteur/workflows/Debruiteur%20CI/badge.svg)

# LeDebruiteur

Computer vision project done during the last Bachelor semester.  
The aim is to show the use of a few filtering methods to reduce different kind of noises.  
And compare deep learning models with more conventional filters.

## Authors :

* Jonas Freiburghaus [@JonasFreibur](https://github.com/JonasFreibur)
* Romain Capocasale [@RomainCapo](https://github.com/RomainCapo)

## Filters

* Sharpness and edge
  * Wiener
  * Laplacian
  * High pass using FFT
  * Gaussian weighted
* Noise
  * Mean
  * Median
  * Low pass using FFT
  * Conservative
  * Gaussian

## Inspired from :

* [Image De-raining Using a Conditional Generative Adversarial Network](https://arxiv.org/pdf/1701.05957.pdf)

## Credits :

To train the models we used the Calthech101 Dataset.

```
L. Fei-Fei, R. Fergus and P. Perona. One-Shot learning of object
categories. IEEE Trans. Pattern Recognition and Machine Intelligence. In
press
```
