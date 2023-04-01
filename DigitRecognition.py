from PIL import Image
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=5)
clf = GaussianNB()
clf.fit(X_train, y_train)
predicted = clf.predict(X_test)
expected = y_test
true = 0
false = 0
fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
for i in range(64):
    ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
    ax.imshow(X_test.reshape(-1, 8, 8)[i], cmap=plt.cm.binary, interpolation='nearest')

    if expected[i] == predicted[i]:
        true+=1
        plt.text(1, 7, str(expected[i]))
        plt.text( 6, 7, str(predicted[i]), color="green")
    else:
        plt.text(1, 7, str(expected[i]))
        plt.text( 6, 7, str(predicted[i]), color="red")
        false+=1
print(true, false)
print(X_train.shape)
plt.show()
