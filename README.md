# pyyolo
pyyolo is a simple wrapper for YOLO.

## Building
1. git clone --recursive https://github.com/oulutan/pyyolo
2. Edit Makefile to use GPU.
3. make
4. python setup.py build
5. sudo python setup.py install

## Test
Edit parameters in example.py
```bash
python example.py
```
Result
```bash
{'right': 194, 'bottom': 353, 'top': 264, 'class': 'dog', 'prob': 0.8198755383491516, 'left': 71}
{'right': 594, 'bottom': 338, 'top': 109, 'class': 'horse', 'prob': 0.6106302738189697, 'left': 411}
{'right': 274, 'bottom': 381, 'top': 101, 'class': 'person', 'prob': 0.702547550201416, 'left': 184}
{'right': 583, 'bottom': 347, 'top': 137, 'class': 'sheep', 'prob': 0.7186083197593689, 'left': 387}
```
