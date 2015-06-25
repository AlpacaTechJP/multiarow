# Multi-class AROW

This is a simple, mimimalist AROW implementation in python.

## Install

```
$ pip install multiarow
```

## Usage

It is super-simple.

```python
import multiarow
# create an instance
arow = multiarow.AROW()

# train
arow.train('label1', np.array([1,2,3], dtype=np.float))
arow.train('label2', np.array([-1,-2,-3], dtype=np.float))

# classify
predicted = arow.classify(np.array([4,5,9], dtype=np.float))
=> 'label1'

# list up previously trained class labels
labels = arow.list_label()
=> ['label1', 'label2']

# delete a class label
arow.delete_label(label1)

```

If you need to persist the current learning result, you can simply use pickle or joblib.
