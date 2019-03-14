# LT2212 V19 Assignment 3

From Asad Sayeed's statistical NLP course at the University of Gothenburg.

My name:    Elin Hagman\
GitHub:     eliyetres\
GU:         gusihaliel

## Additional instructions

Use the -T option to select n number of lines from the training data to be used as test data.

## Reporting for Part 4

The hypothesis is that perplexity decreases and accuracy increases with higher n-grams and a larger data set.

Results

| Arguments                | Train/test ratio | N-grams | Accuracy            | Perplexity         |
|--------------------------|------------------|---------|---------------------|--------------------|
| *Larger data sets*         |                  |         |                     |                    |
| -S 500 -E 900 -T 60      | 300/100          | 3       | 0.12033898305084746 | 63.66197394489233  |
| -N 4 -S 500 -E 900 -T 60 | 300/100          | 4       | 0.1211864406779661  | 63.95347533991584  |
| -N 5 -S 500 -E 900 -T 60 | 300/100          | 5       | 0.44710947109471094 | 52.71499737446524  |
| -N 6 -S 500 -E 900 -T 60 | 300/100          | 6       | 0.10932203389830508 | 66.41186261008907  |
| *Smaller data set*         |                  |         |                     |                    |
| -S 500 -E 536 -T 6       | 30/10            | 3       | 0.3176470588235294  | 31.0006128203334   |
| -N 4 -S 500 -E 536 -T 6  | 30/10            | 4       | 0.09868421052631579 | 33.771870329218345 |
| -N 5 -S 500 -E 536 -T 6  | 30/10            | 5       | 0.8067226890756303  | 25.955131727682794 |
| -N 6 -S 500 -E 536 -T 6  | 30/10            | 6       | 0.9831932773109243  | 22.478650040799288 |
|                          |                  |         |                     |                    |

The results show that accuracy increases with a larger data set, but that perplexity increases with a larger data set.
There is only a slight increase in accuracy with higher n-grams but also a slight increase in perplexity.

When using 5-grams for the 300/60 set the data has the highest accuracy and the lowest perplexity ( don't know if this is correct, might have to run this part again).
