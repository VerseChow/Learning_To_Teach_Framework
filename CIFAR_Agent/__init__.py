import os, sys
lib_path = os.path.abspath(os.path.join('..'))
sys.path.append(lib_path)
__all__ = ['cifar_train', 'model']