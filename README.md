# Chest X-Ray Pneumonia Detection using Deep Learning (PyTorch)

Ένα ολοκληρωμένο Machine Learning project για την αυτόματη ανίχνευση πνευμονίας από ακτινογραφίες θώρακα (Chest X-Rays). Το project υλοποιήθηκε σε περιβάλλον **Google Colab με χρήση T4 GPU (CUDA)** για την ταχεία εκπαίδευση του μοντέλου.

## 📊 Σύνολο Δεδομένων (Dataset)
Χρησιμοποιήθηκε το ευρέως αναγνωρισμένο ιατρικό dataset **Chest X-Ray Images (Pneumonia)** από το Kaggle. Αποτελείται από χιλιάδες ψηφιοποιημένες ακτινογραφίες θώρακα, χωρισμένες σε κατηγορίες `NORMAL` (Υγιείς) και `PNEUMONIA` (Ασθενείς). 

*Σημείωση: Λόγω της ανισορροπίας στη διανομή του αρχικού validation set, το Test Loader χρησιμοποιήθηκε ως βασικό σύνολο ελέγχου κατά τη διαδικασία αξιολόγησης.*

## 🛠️ Τεχνολογίες & Εργαλεία
* **Γλώσσα Προγραμματισμού:** Python
* **Deep Learning Framework:** PyTorch & Torchvision
* **Data Evaluation:** Scikit-Learn (Classification Report, Confusion Matrix), Numpy
* **Hardware Accelerator:** Google Colab T4 GPU (CUDA Compute)

## 🧠 Αρχιτεκτονική Μοντέλου & Εκπαίδευση
Εφαρμόστηκε η τεχνική του **Transfer Learning** με τη χρήση του προ-εκπαιδευμένου νευρωνικού δικτύου **ResNet18**. 
1. Όλα τα αρχικά επίπεδα (feature extractors) του ResNet18 "παγώθηκαν" (`requires_grad = False`) ώστε να διατηρηθεί η υπάρχουσα γνώση αναγνώρισης σχημάτων.
2. Το τελευταίο Fully Connected επίπεδο (`model.fc`) αντικαταστάθηκε με ένα νέο γραμμικό επίπεδο 2 εξόδων (Normal vs Pneumonia).
3. Για την εκπαίδευση χρησιμοποιήθηκε ο αλγόριθμος **Adam Optimizer** (με $lr=0.001$) και η συνάρτηση απώλειας **CrossEntropyLoss**.

## 📈 Ιατρικά Αποτελέσματα & Αξιολόγηση (Metrics)
Σε ιατρικές εφαρμογές, το πιο κρίσιμο metric είναι το **Recall (Sensitivity)** της θετικής κλάσης (`PNEUMONIA`), καθώς στόχος είναι η ελαχιστοποίηση των False Negatives (ασθενείς που νοσούν αλλά διαγιγνώσκονται λανθασμένα ως υγιείς).
Το μοντέλο πέτυχε κορυφαίο Pneumonia Recall της τάξης του 97% και συνολική ακρίβεια (Accuracy) 88% από την πρώτη κιόλας εποχή εκπαίδευσης.

Κατά τη διάρκεια της δεύτερης εποχής εφαρμόστηκε στρατηγική έγκαιρου τερματισμού (Early Stopping). Παρατηρήθηκε ότι ενώ το Train Loss μειωνόταν περαιτέρω (στο 0.1393) και το Train Accuracy άγγιξε το 94.42%, το Validation Loss παρουσίασε άνοδο (στο 0.4939). Αυτό το ξεκάθαρο σημάδι Overfitting (υπερεκπαίδευσης) αντιμετωπίστηκε με τη διατήρηση των βαρών της 1ης εποχής, διασφαλίζοντας ότι το μοντέλο διατηρεί τη βέλτιστη ικανότητα γενίκευσης (generalization) σε νέα, άγνωστα δεδομένα.

### Classification Report
```text
              precision    recall  f1-score   support

      NORMAL       0.93      0.73      0.82       234
   PNEUMONIA       0.86      0.97      0.91       390

    accuracy                           0.88       624
   macro avg       0.90      0.85      0.86       624
weighted avg       0.89      0.88      0.88       624




True Negative (Υγιείς που βρέθηκαν Υγιείς): 171
False Positive (Υγιείς που βρέθηκαν με Πνευμονία): 63
False Negative (Ασθενείς που βρέθηκαν Υγιείς -> ΚΡΙΣΙΜΟ ΛΑΘΟΣ): 12
True Positive (Ασθενείς που βρέθηκαν με Πνευμονία): 378



