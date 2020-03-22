# Agree-to-Disagree (A2D): A Deep Learning based Framework for Authorship Discrimination Task in Corpus-specificity Free Manner

In this work, we propose Agree-to-Disagree (A2D), a novel framework for the authorship discrimination task. It is a two-stage deep learning-based framework consisting of an 'Agree' and a 'Disagree' network. At the first stage, it learns the authorship attributes with its Agree network. Subsequently, through its Disagree network, the framework attempts to differentiate the authorship of a new dataset. We perform our experiments on two benchmark datasets (Reuter_C50, Spooky_Author), having contents from multiple authors, and our framework achieves impressive results for the author discrimination task on those datasets. We show that A2D is not dependent on the dataset-specific prior knowledge and it can learn only from authorship attributes of the dataset to detect whether two different writings are from the same author. We prove that the A2D framework can successfully reveal the authorship with pseudonyms through tasking it with unfolding the pseudonyms of a famous American short story writer Washington Irving. We also apply our framework on a historical topic of ascertaining whether the authorship of the most respected book in Islam (the Holy Quran) can be attributed to the Prophet of Islam. Through the experimental analysis, A2D reveals that the Prophet of Islam is not the author of the Holy Quran, and this result is in perfect alignment with the belief of 1.8 billion Muslims around the globe regarding the authorship of this holy book.



## Requirements

* Python 3.6/Anaconda
* Numpy
* Matplotlib 
* Pandas
* Keras
* Google Colaboratory/Ipynb
