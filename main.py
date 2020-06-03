from tree import evaluate_algorithm
from tree import load_csv_as_matrix
from tree import decision_tree
filename = 'train.csv'
dataset = load_csv_as_matrix(filename)
dataset.pop(0)
dataset = dataset[0:]

for row in dataset:
	if int(row[len(row)-1])>=6:
		row[len(row)-1] = 1
	else:
		row[len(row)-1] = 0
testset = load_csv_as_matrix('test.csv')
testset.pop(0)
for row in testset:
	if int(row[len(row)-1])>=6:
		row[len(row)-1] = 1
	else:
		row[len(row)-1] = 0
n_folds = 12
max_depth = 45
min_size = 1

print('Start!-----------')
scores = evaluate_algorithm(dataset,testset,decision_tree, n_folds, max_depth, min_size)
print('Scores: %s' % scores)
print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
