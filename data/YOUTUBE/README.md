# Content Description

* For description of {d,U,validation,test}processed.p see the main README of this repository
* 01_spam_tutorial.py  generates the data
	- This file is a modified version of [this](https://github.com/snorkel-team/snorkel-tutorials/blob/master/spam/01_spam_tutorial.ipynb) file in [snorkel-tutorials repository](https://github.com/snorkel-team/snorkel-tutorials) 
	- our changes appear from line 708 in 01_spam_tutorial.py
	- these changes simply dump the data in our format

# Note:
* Since sentences are randomly distributed into train test and valid files, the pickle files you may generate might be little different than what is dumped here.
* If you are re-generating the pickles using 01_spam_tutorial.py, then you should re-train the snorkel's saved_label_model using run_snorkel.py (See the main README)
* Pickle files dumped here were used in our experiments.

