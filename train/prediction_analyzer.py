import numpy as np

class PredictionAnalysis():
    def __init__(self, y_true, y_pred, dataset, reversed_original_mapping, reversed_virtual_mapping):
        self.y_true, self.y_pred = np.array(y_true), np.array(y_pred)
        self.dataset = dataset
        self.original_mapping, self.virtual_mapping = reversed_original_mapping, reversed_virtual_mapping

    def analyze_misclassified_instances(self, batch_num):
        new_classes, old_classes = [[self.original_mapping[item] for item in each] for each in
                                    (self.dataset.classes_by_groups[batch_num],
                                     [item for each in self.dataset.classes_by_groups[:batch_num] for item in each])]
        misclassified_new_classes, total_new_classes = self.get_wrongly_predicted_count(new_classes)
        if batch_num > 0:
            misclassified_old_classes, total_old_classes = self.get_wrongly_predicted_count(old_classes)
            old_misclassified_old, old_misclassified_new = self.get_wrongly_predicted_old_classes(old_classes, new_classes)
            result = (misclassified_new_classes, total_new_classes, misclassified_old_classes, total_old_classes, old_misclassified_new, old_misclassified_old)
        else:
            result = (misclassified_new_classes, total_new_classes)
        return result

    def get_wrongly_predicted_count(self, true_class_labels):
        true_label_indices = np.where(np.in1d(self.y_true, true_class_labels))[0]
        filtered_y_true, filtered_y_pred = self.y_true[true_label_indices], self.y_pred[true_label_indices]
        array_of_equals = np.equal(filtered_y_true, filtered_y_pred)
        return array_of_equals.shape[0] - sum(array_of_equals), filtered_y_true.shape[0]

    def get_wrongly_predicted_old_classes(self, old_class_labels, new_class_labels):
        old_labels_indices = np.where(np.in1d(self.y_true, old_class_labels))[0]
        filtered_y_true, filtered_y_pred = self.y_true[old_labels_indices], self.y_pred[old_labels_indices]

        old_misclassified_old = [i for i,j in zip(filtered_y_true, filtered_y_pred) if j in old_class_labels and j != i]
        old_misclassified_new = [i for i,j in zip(filtered_y_true, filtered_y_pred) if j in new_class_labels]
        return len(old_misclassified_old), len(old_misclassified_new)




