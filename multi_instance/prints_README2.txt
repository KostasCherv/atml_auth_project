Πληροφορίες σχετικά με τις εκτυπώσεις του κώδικα:

Στην αρχή εκτυπώνονται αριθμοί 1,2,3... που αντιστοιχούν σε κάθε εικόνα που έχει μετατραπεί σε bag, μέχρις ότου να μετατραπούν όλες οι εικόνες.

Τα πρώτα τέσσερα πακέτα εκτυπώσεων των μετρήσεων Accuracy, Precision, Recall και F1 
αντιστοιχούν στα αποτελέσματα των SVM, KNN, DTRE και NN, αντίστοιχα, αφού έχουν μετατραπεί τα δεδομένα
με τη μέθοδο kmeans.

Το πέμπτο πακέτο εκτυπώσεων Accuracy, Precision, Recall και F1
αντιστοιχεί στα αποτελέσματα του citation-knn με τα δεδομένα ως bags.

Το έκτο πακέτο εκτυπώσεων Accuracy, Precision, Recall και F1
αντιστοιχεί στα αποτελέσματα του misvm για τρεις κλάσεις (1vs1), με τα δεδομένα ως bags.

Τα παραπάνω αποτελέσματα αντιστοιχούν στο dataset και τον generator που θα επιλέξει ο χρήστης.

Το έβδομο πακέτο εκτυπώσεων Accuracy, Precision, Recall και F1
αντιστοιχεί στα αποτελέσματα του misvm για τα binary δεδομένα του δεύτερου dataset, με τα δεδομένα ως bags.

Στην αναφορά και στην παρουσίαση παρουσιάζονται και τα αποτελέσματα του citation-knn για τα binary δεδομένα του 
δεύτερου dataset. Ωστόσο, επειδή πραγματοποιήθηκε σε μεταγενέστερο χρόνο για σύγκριση με τα αντίστοιχα αποτελέσματα του misvm
εκ παραδρομής δεν εντάσσεται η αντίστοιχη υλοποίηση στο αρχείο multi_instance_leaarning.py
Η υλοποίηση του μπορεί να υλοποιηθεί εφαρμόζοντας τις παρακάτω εντολές στο τέλος του κώδικα:


X_train, X_test, y_train, y_test = model_selection.train_test_split(XX, yy, test_size = 0.25, random_state = 0)
model= CitationKNN()
model.fit(X_train, y_train)
y_predicted3 = model.predict(X_test)
y_test3 = np.array(y_test, dtype = np.float)
print("Accuracy:  %9f" % metrics.accuracy_score(y_test3, y_predicted3))
print("Precision:  %2f" % metrics.precision_score(y_test3, y_predicted3, average='macro'))
print("Recall:  %11f" % metrics.recall_score(y_test3, y_predicted3, average='macro'))
print("F1:  %15f \n" % metrics.f1_score(y_test3, y_predicted3, average='macro'))

Έτσι το τελευταίο πακέτο εκτυπώσεων δείχνει τα αποτελέσματα του citation-knn στα binary δεδομένα του δεύτερου dataset.
