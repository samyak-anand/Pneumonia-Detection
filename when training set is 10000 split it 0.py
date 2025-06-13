when training set is 10000 split it 0.2 

---------------------------------------------------------------------------
MemoryError                               Traceback (most recent call last)
Cell In[93], line 2
      1 # Convert grayscale to RGB if needed
----> 2 train_X_rgb = np.repeat(train_X[..., np.newaxis], 3, -1)

File c:\Users\samya\PyCharmProject\Pneumonia-Detection\.venv\Lib\site-packages\numpy\_core\fromnumeric.py:511, in repeat(a, repeats, axis)
    467 @array_function_dispatch(_repeat_dispatcher)
    468 def repeat(a, repeats, axis=None):
    469     """
    470     Repeat each element of an array after themselves
    471 
   (...)    509 
    510     """
--> 511     return _wrapfunc(a, 'repeat', repeats, axis=axis)

File c:\Users\samya\PyCharmProject\Pneumonia-Detection\.venv\Lib\site-packages\numpy\_core\fromnumeric.py:57, in _wrapfunc(obj, method, *args, **kwds)
     54     return _wrapit(obj, method, *args, **kwds)
     56 try:
---> 57     return bound(*args, **kwds)
     58 except TypeError:
     59     # A TypeError occurs if the object does have such a method in its
     60     # class, but its signature is not identical to that of NumPy's. This
   (...)     64     # Call _wrapit from within the except clause to ensure a potential
     65     # exception has a traceback chain.
     66     return _wrapit(obj, method, *args, **kwds)

MemoryError: Unable to allocate 3.66 GiB for an array with shape (10000, 128, 128, 3) and data type float64
     
---------------------------------------------------------------------------
MemoryError                               Traceback (most recent call last)
Cell In[84], line 1
----> 1 hsitory_cnn= model_cnn.fit(train_X_rgb, train_Y,
      2                            batch_size= 32,
      3                            epochs=50,
      4                            class_weight = classweight, 
      5                            validation_split = 0.2 
      6                            )

File c:\Users\samya\PyCharmProject\Pneumonia-Detection\.venv\Lib\site-packages\keras\src\utils\traceback_utils.py:122, in filter_traceback.<locals>.error_handler(*args, **kwargs)
    119     filtered_tb = _process_traceback_frames(e.__traceback__)
    120     # To get the full stack trace, call:
    121     # `keras.config.disable_traceback_filtering()`
--> 122     raise e.with_traceback(filtered_tb) from None
    123 finally:
    124     del filtered_tb

File c:\Users\samya\PyCharmProject\Pneumonia-Detection\.venv\Lib\site-packages\optree\ops.py:766, in tree_map(func, tree, is_leaf, none_is_leaf, namespace, *rests)
    764 leaves, treespec = _C.flatten(tree, is_leaf, none_is_leaf, namespace)
    765 flat_args = [leaves] + [treespec.flatten_up_to(r) for r in rests]
--> 766 return treespec.unflatten(map(func, *flat_args))

MemoryError: Unable to allocate 1.83 GiB for an array with shape (10000, 128, 128, 3) and data type float32