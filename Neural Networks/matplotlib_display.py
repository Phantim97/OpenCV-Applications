import matplotlib.pyplot as plt

with open("loss_acc_train.txt",'r') as train_file:
    train_data = train_file.readlines()

with open("loss_acc_test.txt",'r') as test_file:
    test_data = test_file.readlines()

train_loss_array = []
train_acc_array = []

for item in train_data:
  loss,acc = item.strip().split(',')
  train_loss_array.append(float(loss))
  train_acc_array.append(float(acc))

test_loss_array = []
test_acc_array = []

for item in test_data:
  loss,acc = item.strip().split(',')
  test_loss_array.append(float(loss))
  test_acc_array.append(float(acc))

print(train_loss_array)

plt.plot(train_loss_array,'b')
plt.plot(test_loss_array,'r')
plt.title("Loss Curves");

plt.plot(train_acc_array,'b')
plt.plot(test_acc_array,'r')
plt.title("Accuracy Curves");