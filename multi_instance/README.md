*  Το dataset 1 μπορεί να ληφθεί από την εξής ηλ. 
  δ/νση: https://www.kaggle.com/twopothead/miml-image-data

* Το dataset 2 μπορεί να ληφθεί από την εξής ηλ. δ/νση: https://md-datasets-public-files-prod.s3.eu-west-1.amazonaws.com/64a54851-f95a-439d-aff7-3334a684e532
(Από το δεύτερο dataset πρέπει να αφαιρεθούν οι εικόνες rain.141 και shine.131)


* Για την εφαρμογή του misvm πρέπει να εγκατασταθούν οι συγκεκριμένες βιβλιοθήκες: 
(προ-απαιτούμενες)

  * ``` pip install numpy ```
  * ``` pip install scipy ```
  * ``` pip install cvxopt ```
  * ```pip install -e git+https://github.com/garydoranjr/misvm.git#egg=misvm```


* Το αρχείο CitationKKN.py είναι αυτό που υλοποιεί τον citation-kNN και λήφθηκε από την εξής ηλ. δ/νση: https://github.com/arjunssharma/Citation-KNN
Ωστόσο στο αρχείο που συγκαταλέγεται εφαρμόστηκαν οι εξής αλλαγές στις παρακάτω σειρές σε σχέση με τον αρχικό κώδικα που βρίσκεται στην παραπάνω δ/νση:

  * Γραμμή 17:  self._no_of_references = kwargs['references'] --> self._no_of_references = 2
  * Γραμμή 18:  self._no_of_citers = kwargs['citers']  --> self._no_of_citers = 4
  Στις οποίες ορίζονται τα references και citers

  * Γραμμή 81:  relevant_test_labels.append([references[j]][0]) --> relevant_test_labels.append(self._labels[[references[j]][0]])
  * Γραμμή 84:  relevant_test_labels.append([citers[j]][0]) --> relevant_test_labels.append(self._labels[[citers[j]][0]])


* Ορισμένες από τις λειτουργίες του κώδικα που περιέχονται στο αρχείο multi_instance_learning.py
περιγράφονται εντός της αναφοράς, όπως επίσης πληροφορίες σχετικά με τα dataset καθώς 
και για τις βιβλιοθήκες που χρησιμοποιήθηκαν για τον citation-kNN και misvm.

Τα path που πρέπει να αλλάξουν στα καινούργια, όπου θα περιέχονται τα αντίστοιχα dataset
για τη λειτουργία του κώδικα βρίσκονται στις εξής σειρές του αρχείου multi_instance_learning.py. 

* Για το πρώτο dataset, στις σειρές: 378, 678, 1086
* Για το πρώτο dataset, στις σειρές: 421, 720


#### Επιπλέον βιβλιοθήκες που πρέπει να εγκατασταθούν:
* ```pip install opencv-python (cv2)```
* ```pip install -U scikit-learn (sklearn)```