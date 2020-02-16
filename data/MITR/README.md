# Content Description

* For description of {d,U,validation,test}processed.p see the main README of this repository
* data_files directory contains the training and testing data downloaded from [here](https://groups.csail.mit.edu/sls/downloads/restaurant)
	- {all.words.txt,all.tags.txt} : Unrolled version of [restauranttrain.bio](https://groups.csail.mit.edu/sls/downloads/restaurant/restauranttrain.bio)
	- {test.words.txt,test.tags.txt}: Unrolled version of [restauranttest.bio](https://groups.csail.mit.edu/sls/downloads/restaurant/restauranttest.bio)
	- {rule_center.words.txt,rule_center.tags.txt}: Words and tags corresponding to sentences inspecting which rules were defined.
* rules.py contains rules defined by observing a few sentences randomly fectched from data_files/all.words.txt
* Remaing sentences in  were randomly distributed into training (used for U set), and testing data.
* generate_data.py creates {d,U,validation,test}processed.p using rules.py
	- get_elmo_contextual_embeddings.py is used, which uses [elmo](https://tfhub.dev/google/elmo/) to obtain contextual embeddings of words in each sentence

# Note:
* Since sentences in training data was randomly distributed into train (U) and test, the pickle files you may generate might be little different than what is dumped here.
* If you are re-generating the pickles using generate_data.py, then you should re-train the snorkel's saved_label_model using run_snorkel.py (See the main README)
* Pickle files dumped here were used in our experiments.

