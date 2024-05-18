# HW2: Implementing Hidden Markov Model (HMM) in Python
## Purpose
The purpose of this assignment is to implement a Hidden Markov Model (HMM) using Python and to train and evaluate it on sequences of English alphabet letters.


## Detailed Task
1. Given Elements:
   * Observation symbols set
   * States set
   * State transition probabilities
   *  Initial state probabilities
   *  Observation distribution function for each state

2. Training Data:
   * Five independent datasets are provided for training five separate Hidden Markov Models (HMMs).

3. Training:
   * Train the five models using the Baum-Welch algorithm, which is based on the Expectation-Maximization (EM) algorithm.

4. Evaluation:
    * Evaluate the trained models on two test datasets using the Viterbi algorithm.
    * For each sample in the test datasets, determine which model maximizes the observation probability.

