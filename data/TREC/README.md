# Content Description

* For description of {d,U,validation,test}processed.p see the main README of this repository
* Original [train](https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label) and [test](https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label) data was obtained from [here](https://cogcomp.seas.upenn.edu/Data/QA/QC/)
* rule.txt contains rules defined by observing a few sentences randomly fectched from original training data
* Format of rule.txt
	- First column represents the label provided by rule
	- Second column represents the regex corresponding to the rule
	- Third column represents the sentence inspecting which rule was designed.
* Remaing sentences in original training data were randomly distributed into train.txt, valid.txt. valid.txt has 500 sentences
* generate_data.py creates {d,U,validation,test}processed.p using rules.txt,train.txt,valid.txt,test.txt respectively
	- obtain_embeddings.py is used, which uses [elmo](https://tfhub.dev/google/elmo/) to convert sentences into embeddings

# Note:
* Since sentences in original training file were randomly distributed into train and valid files, the pickle files you may generate might be little different than what is dumped here.
* If you are re-generating the pickles using generate_data.py, then you should re-train the snorkel's saved_label_model using run_snorkel.py (See the main README)
* Pickle files dumped here were used in our experiments.

