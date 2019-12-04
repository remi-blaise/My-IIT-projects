BERT vectors for text classification
====================================

By Rémi Blaise for the course "CS 585: Natural Language processing", by Derrick Higgings at the Illinois Institute of 
Technology.


## Approach

In this project I have compared 6 different models from the scikit-learn library with 104 combinations of parameters.
I use the accuracy computed on the `test` dataset to explore the parameter space of each model, then I evaluate the 
obtained model by computing the accuracy on the `eval` dataset.


## Implementation details

- In order to have shorter execution time while training and running the classifiers, the feature reader function 
written in `classifiers/read_data.py` cache the data read from the huge BERT's output.
- All classifiers are defined in `classifiers/models.py` and use sckikit-learn algorithms. In order to make them have a
common specification, API and share common code, they inherit from an abstract class `AbstractClassifier` and a base 
class `BaseSklearnClassifier`.
- I enjoyed doing this assignment.


## How to run the code

*All script are made to be executed from the root folder of this project.*

0. Prerequisite: make sure you have all dependencies installed:
```bash
sudo pip3 install numpy pandas scikit-learn PrettyTable
sudo pip3 install tensorflow # or tensorflow-gpu, see https://www.tensorflow.org/install/gpu for installation details.
```

Scripts have been tested using python3.6.

1. Generate BERT input data by running:

```bash
./format.py
``` 

2. Download given [`bert`](https://github.com/google-research/bert) and [`uncased_L-12_H-768_A-12`](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip) folders. Then customize `BERT_BASE_DIR` and `BERT_DATA_DIR` 
variables in `run_bert_fv.sh` to reflect the location of your folders.

3. Generate features by running:

```bash
./run_bert_fv.sh
```

This took me about 12 minutes with tensorflow-gpu running on Nvidia GTX 1050.

4. In the file `sklearn_models.py`, configure the `CORE` variable to use the number of cores you want to exploit (`-1` 
for using all).

5. The following program will compare all 104 explored options of 6 classifier models and give the best:

```bash
classifiers/models.py all
```

This will save the best classifier in the file `best_classifier`.

It took me about **1h15** with an Intel Core i5 9th Gen.

You can explore only one of the models by changing the parameter: 

```bash
classifiers/models.py [LogisticRegressionClassifier|SVMClassifier|RandomForestClassifier|KNeighborsClassifier|
	MultiLayerPerceptronClassifier]
```

I have saved the output in `output.txt` (have a look!) and `best_classifier` so you can go directly to the next step if 
you wish.

6. Print metrics about the predicted data:

```bash
./analyse.py
```


## Results

The best model according to the log file `output.txt` is:

```python
MLPClassifier(hidden_layer_sizes=(100,)*5, solver='adam',  activation='relu', early_stopping=True)
```

It gets an accuracy on the `eval` set of **0.471**, which can seem not so impressive but is much better than a random 
guess (accuracy would be 0.1 given there are 10 classes).

We get the following statistics by classes:

```
+------------+----------------+-----------------+---------------+----------------+----------------+-----------------------+-------------------+---------------+------------+
|   class    | total_expected | total_predicted | true_positive | false_positive | false_negative | most_misclassified_as | precision_measure | recall_mesure | f1_measure |
+------------+----------------+-----------------+---------------+----------------+----------------+-----------------------+-------------------+---------------+------------+
|   Arabic   |      200       |       161       |       83      |       78       |      117       |        Spanish        |       0.516       |     0.415     |   0.460    |
| Cantonese  |      200       |       164       |       67      |       97       |      133       |        Mandarin       |       0.409       |     0.335     |   0.368    |
|  Japanese  |      200       |       336       |      130      |      206       |       70       |        Russian        |       0.387       |     0.650     |   0.485    |
|   Korean   |      200       |       141       |       73      |       68       |      127       |        Japanese       |       0.518       |     0.365     |   0.428    |
|  Mandarin  |      200       |       144       |       59      |       85       |      141       |        Japanese       |       0.410       |     0.295     |   0.343    |
|   Polish   |      200       |       264       |      119      |      145       |       81       |        Russian        |       0.451       |     0.595     |   0.513    |
|  Russian   |      200       |       224       |      117      |      107       |       83       |         Polish        |       0.522       |     0.585     |   0.552    |
|  Spanish   |      200       |       227       |      121      |      106       |       79       |         Polish        |       0.533       |     0.605     |   0.567    |
|    Thai    |      200       |       158       |      114      |       44       |       86       |       Vietnamese      |       0.722       |     0.570     |   0.637    |
| Vietnamese |      200       |       181       |       77      |      104       |      123       |        Japanese       |       0.425       |     0.385     |   0.404    |
+------------+----------------+-----------------+---------------+----------------+----------------+-----------------------+-------------------+---------------+------------+
```

as well as the following number of classifications (mis- and well- classified) between each pair of classes:

```
+--------------------------+----------------------+
|         classes          | number of classified |
+--------------------------+----------------------+
|     Arabic as Arabic     |          83          |
|   Arabic as Cantonese    |          9           |
|    Arabic as Japanese    |          11          |
|     Arabic as Korean     |          13          |
|    Arabic as Mandarin    |          3           |
|     Arabic as Polish     |          17          |
|    Arabic as Russian     |          15          |
|    Arabic as Spanish     |          25          |
|      Arabic as Thai      |          6           |
|   Arabic as Vietnamese   |          18          |
|   Cantonese as Arabic    |          10          |
|  Cantonese as Cantonese  |          67          |
|  Cantonese as Japanese   |          26          |
|   Cantonese as Korean    |          6           |
|  Cantonese as Mandarin   |          40          |
|   Cantonese as Polish    |          14          |
|   Cantonese as Russian   |          13          |
|   Cantonese as Spanish   |          9           |
|    Cantonese as Thai     |          6           |
| Cantonese as Vietnamese  |          9           |
|    Japanese as Arabic    |          5           |
|  Japanese as Cantonese   |          6           |
|   Japanese as Japanese   |         130          |
|    Japanese as Korean    |          12          |
|   Japanese as Mandarin   |          4           |
|    Japanese as Polish    |          9           |
|   Japanese as Russian    |          12          |
|   Japanese as Spanish    |          7           |
|     Japanese as Thai     |          8           |
|  Japanese as Vietnamese  |          7           |
|     Korean as Arabic     |          15          |
|   Korean as Cantonese    |          10          |
|    Korean as Japanese    |          51          |
|     Korean as Korean     |          73          |
|    Korean as Mandarin    |          6           |
|     Korean as Polish     |          9           |
|    Korean as Russian     |          12          |
|    Korean as Spanish     |          5           |
|      Korean as Thai      |          8           |
|   Korean as Vietnamese   |          11          |
|    Mandarin as Arabic    |          10          |
|  Mandarin as Cantonese   |          30          |
|   Mandarin as Japanese   |          37          |
|    Mandarin as Korean    |          6           |
|   Mandarin as Mandarin   |          59          |
|    Mandarin as Polish    |          15          |
|   Mandarin as Russian    |          7           |
|   Mandarin as Spanish    |          12          |
|     Mandarin as Thai     |          2           |
|  Mandarin as Vietnamese  |          22          |
|     Polish as Arabic     |          2           |
|   Polish as Cantonese    |          6           |
|    Polish as Japanese    |          16          |
|     Polish as Korean     |          4           |
|    Polish as Mandarin    |          5           |
|     Polish as Polish     |         119          |
|    Polish as Russian     |          26          |
|    Polish as Spanish     |          19          |
|      Polish as Thai      |          1           |
|   Polish as Vietnamese   |          2           |
|    Russian as Arabic     |          5           |
|   Russian as Cantonese   |          3           |
|   Russian as Japanese    |          16          |
|    Russian as Korean     |          3           |
|   Russian as Mandarin    |          5           |
|    Russian as Polish     |          32          |
|    Russian as Russian    |         117          |
|    Russian as Spanish    |          9           |
|     Russian as Thai      |          4           |
|  Russian as Vietnamese   |          6           |
|    Spanish as Arabic     |          11          |
|   Spanish as Cantonese   |          4           |
|   Spanish as Japanese    |          6           |
|    Spanish as Korean     |          4           |
|   Spanish as Mandarin    |          3           |
|    Spanish as Polish     |          32          |
|    Spanish as Russian    |          8           |
|    Spanish as Spanish    |         121          |
|     Spanish as Thai      |          2           |
|  Spanish as Vietnamese   |          9           |
|      Thai as Arabic      |          5           |
|    Thai as Cantonese     |          18          |
|     Thai as Japanese     |          15          |
|      Thai as Korean      |          8           |
|     Thai as Mandarin     |          8           |
|      Thai as Polish      |          5           |
|     Thai as Russian      |          3           |
|     Thai as Spanish      |          4           |
|       Thai as Thai       |         114          |
|    Thai as Vietnamese    |          20          |
|   Vietnamese as Arabic   |          15          |
| Vietnamese as Cantonese  |          11          |
|  Vietnamese as Japanese  |          28          |
|   Vietnamese as Korean   |          12          |
|  Vietnamese as Mandarin  |          11          |
|   Vietnamese as Polish   |          12          |
|  Vietnamese as Russian   |          11          |
|  Vietnamese as Spanish   |          16          |
|    Vietnamese as Thai    |          7           |
| Vietnamese as Vietnamese |          77          |
+--------------------------+----------------------+
```


## Improvements

In this assignment, I used only the suggested BERT feature making algorithm as well as common scikit-learn integrated 
models. To find better features and models, I would read current literature about the Native Language Identification 
problem and look for the state-of-the-state machine learning.

Also, a better machine or a longer execution time would allow more exploration in the parameter spaces. Exploration 
algorithms like gradient descent could also be used to lead deeper explorations. However, if a deeper exploration could 
improve the tweaking of the parameters, I don't believe it will bring drastic improvements in the accuracy score.
