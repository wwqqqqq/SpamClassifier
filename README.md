# Spam Detector

## Dataset: [SpamAssassin public mail corpus](https://spamassassin.apache.org/old/publiccorpus/readme.html)

* **Spam**: 500 spam messages, all received from non-spam-trap sources.
* **Easy ham**: 2500 non-spam messages. These are typically quite easy to differentiate from spam, since they frequently do not contain any spammish signatures (like HTML etc).

## Metrics
* Use precision and recall
* Precision: $SP = \frac{n_{pam\rightarrow pam}}{n_{pam\rightarrow pam}+n_{normal\rightarrow pam}}$
* Recall: $SR = \frac{n_{pam\rightarrow pam}}{N_{pam}}$
  * $n_{pam\rightarrow pam}$: the number of spam messages that are correctly classified.
  * $n_{normal\rightarrow pam}$: the number of normal messages that are mistakenly classified as spam.
  * $N_{pam}$: the amount of spam messages in the dataset.
* Metric: $\text{F} = \frac{SP\times SR \times 2}{SP+SR}$
  * Larger $\text{F}$ value indicates better model peformance.