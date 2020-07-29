from . import tools
from . import file_processing
from . import image_processing
from .logger_utils import logger
from .timer_utils import timer_wrapper

"""
47.61466383934021
[[[    0 32492 32303 ... 69943 64658 30243]
  [    1 50279 64981 ... 43962 72871 28638]
  [    2 68855 79564 ...  8600 65064 72660]
  ...
  [39997 55399 77551 ... 54029 25751 11168]
  [39998   275 42718 ... 41654 43394  6839]
  [39999  5108 38672 ... 56380 43893 74363]]

 [[    0 33779 20500 ... 15113 12339 17584]
  [    1 50550 76967 ... 34217 29041 66077]
  [    2 15776 23640 ... 54819 72407  5218]
  ...
  [39997 40357  6517 ... 59976 20285 76110]
  [39998 25823 39517 ... 43221 21620  8803]
  [39999 39858 38159 ... 46103 42359 37550]]

 [[    0 23376 31781 ...  4674 39317 65160]
  [    1 43576 80093 ... 80923 39736 60652]
  [    2 25519 24915 ... 75096 78771 70110]
  ...
  [39997 28974 80526 ...  6238 17100 26394]
  [39998 74603  3317 ... 57853 47591 74937]
  [39999 71967 14852 ... 68960 57578 32300]]

 ...

 [[    0   252   337 ...  1597  6847  3677]
  [    1   429   216 ...  1983  4098  3874]
  [    2  5104  6674 ...    61  1915 10428]
  ...
  [39997 79752 42940 ... 80656 77684 77839]
  [39998 40960 42191 ... 36672 34383 32857]
  [39999 39214 31129 ... 22858 37113 35122]]

 [[    0 78634 78835 ... 75717 77530 74932]
  [    1 57593 58220 ... 62370 48578 49354]
  [    2   701  1577 ...  4096  4709  3689]
  ...
  [39997 39895 44914 ... 64617 65115 20676]
  [39998 40985 38943 ... 37036 38237 38849]
  [39999 39428 39980 ... 38224 41118 41152]]

 [[    0 38970 39159 ... 28795 25005 72193]
  [    1 41598  1308 ...  2006  2412  2994]
  [    2   905  5590 ... 22814 21779 30120]
  ...
  [39997 38222 74911 ... 81462 80835 79408]
  [39998 77793 76557 ... 81463 72074 38708]
  [39999 39117 40069 ... 44010 43372 38547]]]
"""