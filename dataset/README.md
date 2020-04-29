create soft link to the "npz" data here

```shell script
ln -s /full/path/to/data.npz /full/path/to/project/dataset/data.npz
```

then you can load data by:

```python
import numpy as np
file = np.load('./dataset/data.npz')
```