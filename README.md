# Spam Detector

## Dataset: [SpamAssassin public mail corpus](https://spamassassin.apache.org/old/publiccorpus/readme.html)

* **Spam**: 500 spam messages, all received from non-spam-trap sources.
* **Easy ham**: 2500 non-spam messages. These are typically quite easy to differentiate from spam, since they frequently do not contain any spammish signatures (like HTML etc).

## Metrics
* Use precision and recall
* Precision: <img src="http://latex.codecogs.com/gif.latex?SP=\frac{n_{pam\,\rightarrow\,pam}}{n_{pam\,\rightarrow\,pam}+n_{normal\,\rightarrow\,pam}}" />
* Recall: <img src="http://latex.codecogs.com/gif.latex?SR=\frac{n_{pam\,\rightarrow\,pam}}{N_{pam}}" />
  * <img src="http://latex.codecogs.com/gif.latex?n_{pam\,\rightarrow\,pam}" />: the number of spam messages that are correctly classified.
  * <img src="http://latex.codecogs.com/gif.latex?n_{normal\,\rightarrow\,pam}" />: the number of normal messages that are mistakenly classified as spam.
  * <img src="http://latex.codecogs.com/gif.latex?N_{pam}" />: the amount of spam messages in the dataset.
* Metric: <img src="http://latex.codecogs.com/gif.latex?\text{F}=\frac{SP\,\times\,SR\,\times\,2}{SP+SR}" />

  * Larger <img src="http://latex.codecogs.com/gif.latex?\text{F}" /> value indicates better model peformance.

## Procedure
1. Pre-processing
   * Remove the header of emails
   * Construct the feature corpus by first extracting keywords from the emails and then quantifying the frequency of the words in database to calculate feature vectors of each email.
2. Train the classifier
3. Test the performance of each model
   * Calculate F1 score.
4. Use cross validation to evaluate and select the best parameters for each model.

## Naive Bayes Classifier
Baye's Theorem: <img src="http://latex.codecogs.com/gif.latex?P(A|B)=\frac{P(B|A)P(A)}{P(B)}" />
* where A and B are events and <img src="http://latex.codecogs.com/gif.latex?P(B)\neq0}" />.
* the probabilities of events A and B are independent.

Emails consist of words. And there are some unique words that indicate higher probability of a message with these words in it, to be a spam.

According to the Baye's Theorem, if we assume that the probability of each word to appear in a message is independent from each other, then:

<img src="http://latex.codecogs.com/gif.latex?P(Spam|Word_1,Word_2,\dots,Word_n)=\frac{P(Word_1|Spam)P(Word_2|Spam)\dots\,P(Word_n|Spam)}{P(Word_1)P(Word_2)\dots\,P(Word_n)}" />

<img src="http://latex.codecogs.com/gif.latex?P(Word_i|Spam)" /> and <img src="http://latex.codecogs.com/gif.latex?P(Word_i)" /> can be easily calculated based on statistical result of the training database. Through these known probabilities, we can calculate how likely a new unlabeled message is a spam.

## Least Square Classifier

## Support Vector Machines