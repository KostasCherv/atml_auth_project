def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='latin1')
    return dict

def load_cifar100(path='./cifar-100-python', target_labels='fine_labels'):
  test_data = unpickle(path + '/test')
  train_data = unpickle(path + '/train') 
  
  X_train = train_data['data']
  y_train = train_data[target_labels]
  X_test = test_data['data']
  y_test = test_data[target_labels]

  return X_train, X_test, y_train, y_test