# TSAI-Assignment-6
### Objective
Achieve 99.4% on test set with using ~8000 parameters under 15 epochs

### Models Overview
This assignment includes three different model architectures implemented in PyTorch, each trained on the MNIST dataset. The models are defined in model_1.py, model_2.py,and model_3.py.

#### Model 1 (model_1.py)
Number of Parameters: 37M
Train Set Accuracy: 100%
Test Set Accuracy: 98.89%
**Training Logs**
EPOCH: 0
Loss=0.0023657511919736862 Batch_id=937 Accuracy=95.23: 100%|█| 938/938 [02:24<0


Test set: Average loss: 0.0561, Accuracy: 9821/10000 (98.21%)

EPOCH: 1
Loss=0.003380001289770007 Batch_id=937 Accuracy=98.35: 100%|█| 938/938 [02:28<00


Test set: Average loss: 0.0404, Accuracy: 9861/10000 (98.61%)

EPOCH: 2
Loss=0.005407964810729027 Batch_id=937 Accuracy=98.85: 100%|█| 938/938 [02:16<00


Test set: Average loss: 0.0387, Accuracy: 9872/10000 (98.72%)

EPOCH: 3
Loss=0.010393925942480564 Batch_id=937 Accuracy=99.10: 100%|█| 938/938 [02:15<00


Test set: Average loss: 0.0418, Accuracy: 9855/10000 (98.55%)

EPOCH: 4
Loss=0.014830994419753551 Batch_id=937 Accuracy=99.42: 100%|█| 938/938 [02:20<00


Test set: Average loss: 0.0429, Accuracy: 9856/10000 (98.56%)

EPOCH: 5
Loss=0.0009368071332573891 Batch_id=937 Accuracy=99.54: 100%|█| 938/938 [02:18<0


Test set: Average loss: 0.0452, Accuracy: 9862/10000 (98.62%)

EPOCH: 6
Loss=0.04085760936141014 Batch_id=937 Accuracy=99.68: 100%|█| 938/938 [02:31<00:


Test set: Average loss: 0.0436, Accuracy: 9885/10000 (98.85%)

EPOCH: 7
Loss=0.006714309565722942 Batch_id=937 Accuracy=99.73: 100%|█| 938/938 [02:17<00


Test set: Average loss: 0.0454, Accuracy: 9870/10000 (98.70%)

EPOCH: 8
Loss=0.005669977981597185 Batch_id=937 Accuracy=99.80: 100%|█| 938/938 [02:15<00


Test set: Average loss: 0.0496, Accuracy: 9878/10000 (98.78%)

EPOCH: 9
Loss=0.0005120097193866968 Batch_id=937 Accuracy=99.87: 100%|█| 938/938 [02:21<0


Test set: Average loss: 0.0477, Accuracy: 9882/10000 (98.82%)

EPOCH: 10
Loss=0.002046323847025633 Batch_id=937 Accuracy=99.92: 100%|█| 938/938 [02:14<00


Test set: Average loss: 0.0537, Accuracy: 9882/10000 (98.82%)

EPOCH: 11
Loss=0.0006758322706446052 Batch_id=937 Accuracy=99.97: 100%|█| 938/938 [02:16<0


Test set: Average loss: 0.0512, Accuracy: 9878/10000 (98.78%)

EPOCH: 12
Loss=3.7735433124908013e-06 Batch_id=937 Accuracy=100.00: 100%|█| 938/938 [02:18


Test set: Average loss: 0.0522, Accuracy: 9886/10000 (98.86%)

EPOCH: 13
Loss=5.561587386182509e-06 Batch_id=937 Accuracy=100.00: 100%|█| 938/938 [02:18<


Test set: Average loss: 0.0546, Accuracy: 9889/10000 (98.89%)

EPOCH: 14
Loss=0.0005918908282183111 Batch_id=937 Accuracy=100.00: 100%|█| 938/938 [02:18<


Test set: Average loss: 0.0557, Accuracy: 9889/10000 (98.89%)


#### Model 2 (model_2.py)
Number of Parameters: 12,006
Train Set Accuracy: 99.69%
Test Set Accuracy: 98.82%
**Training Logs**
EPOCH: 0
Loss=0.011363556608557701 Batch_id=937 Accuracy=95.62: 100%|█| 938/938 [00:25<00

Test set: Average loss: 0.0650, Accuracy: 9776/10000 (97.76%)

EPOCH: 1
Loss=0.03748512640595436 Batch_id=937 Accuracy=98.31: 100%|█| 938/938 [00:26<00:

Test set: Average loss: 0.0462, Accuracy: 9841/10000 (98.41%)

EPOCH: 2
Loss=0.006760403048247099 Batch_id=937 Accuracy=98.69: 100%|█| 938/938 [00:24<00

Test set: Average loss: 0.0376, Accuracy: 9867/10000 (98.67%)

EPOCH: 3
Loss=0.000871172349434346 Batch_id=937 Accuracy=98.93: 100%|█| 938/938 [00:33<00

Test set: Average loss: 0.0364, Accuracy: 9873/10000 (98.73%)

EPOCH: 4
Loss=0.07656792551279068 Batch_id=937 Accuracy=99.12: 100%|█| 938/938 [00:31<00:

Test set: Average loss: 0.0317, Accuracy: 9897/10000 (98.97%)

EPOCH: 5
Loss=0.0008656036807224154 Batch_id=937 Accuracy=99.22: 100%|█| 938/938 [00:34<0

Test set: Average loss: 0.0398, Accuracy: 9860/10000 (98.60%)

EPOCH: 6
Loss=0.0012317205546423793 Batch_id=937 Accuracy=99.29: 100%|█| 938/938 [00:31<0

Test set: Average loss: 0.0315, Accuracy: 9894/10000 (98.94%)

EPOCH: 7
Loss=0.0032352216076105833 Batch_id=937 Accuracy=99.41: 100%|█| 938/938 [00:32<0

Test set: Average loss: 0.0329, Accuracy: 9885/10000 (98.85%)

EPOCH: 8
Loss=0.017142895609140396 Batch_id=937 Accuracy=99.42: 100%|█| 938/938 [00:32<00

Test set: Average loss: 0.0332, Accuracy: 9884/10000 (98.84%)

EPOCH: 9
Loss=0.1223936378955841 Batch_id=937 Accuracy=99.54: 100%|█| 938/938 [00:31<00:0

Test set: Average loss: 0.0375, Accuracy: 9883/10000 (98.83%)

EPOCH: 10
Loss=0.00138790940400213 Batch_id=937 Accuracy=99.54: 100%|█| 938/938 [00:35<00:

Test set: Average loss: 0.0312, Accuracy: 9907/10000 (99.07%)

EPOCH: 11
Loss=0.0007469068514183164 Batch_id=937 Accuracy=99.57: 100%|█| 938/938 [00:30<0

Test set: Average loss: 0.0306, Accuracy: 9903/10000 (99.03%)

EPOCH: 12
Loss=0.00376885081641376 Batch_id=937 Accuracy=99.67: 100%|█| 938/938 [00:32<00:

Test set: Average loss: 0.0344, Accuracy: 9895/10000 (98.95%)

EPOCH: 13
Loss=0.0022888812236487865 Batch_id=937 Accuracy=99.67: 100%|█| 938/938 [00:32<0

Test set: Average loss: 0.0304, Accuracy: 9905/10000 (99.05%)

EPOCH: 14
Loss=0.00011366191029082984 Batch_id=937 Accuracy=99.69: 100%|█| 938/938 [00:32<

Test set: Average loss: 0.0406, Accuracy: 9882/10000 (98.82%)


#### Model 3 (model_3.py)
Number of Parameters: 7598
Train Set Accuracy: 99.55%
Test Set Accuracy: 99.09%
**Training Logs**
EPOCH: 0
Loss=0.4454112946987152 Batch_id=937 Accuracy=85.51: 100%|█| 938/938 [00:34<00:0

Test set: Average loss: 0.4637, Accuracy: 9495/10000 (94.95%)

EPOCH: 1
Loss=0.3362322449684143 Batch_id=937 Accuracy=95.48: 100%|█| 938/938 [00:34<00:0

Test set: Average loss: 0.2312, Accuracy: 9519/10000 (95.19%)

EPOCH: 2
Loss=0.17509765923023224 Batch_id=937 Accuracy=96.88: 100%|█| 938/938 [00:33<00:

Test set: Average loss: 0.2908, Accuracy: 9102/10000 (91.02%)

EPOCH: 3
Loss=0.1384483128786087 Batch_id=937 Accuracy=97.37: 100%|█| 938/938 [00:36<00:0

Test set: Average loss: 0.0820, Accuracy: 9779/10000 (97.79%)

EPOCH: 4
Loss=0.06538242846727371 Batch_id=937 Accuracy=97.89: 100%|█| 938/938 [00:35<00:

Test set: Average loss: 0.0601, Accuracy: 9824/10000 (98.24%)

EPOCH: 5
Loss=0.05748055502772331 Batch_id=937 Accuracy=98.10: 100%|█| 938/938 [00:33<00:

Test set: Average loss: 0.0682, Accuracy: 9783/10000 (97.83%)

EPOCH: 6
Loss=0.06652931869029999 Batch_id=937 Accuracy=98.31: 100%|█| 938/938 [00:34<00:

Test set: Average loss: 0.0599, Accuracy: 9814/10000 (98.14%)

EPOCH: 7
Loss=0.020944442600011826 Batch_id=937 Accuracy=98.43: 100%|█| 938/938 [00:33<00

Test set: Average loss: 0.0658, Accuracy: 9798/10000 (97.98%)

EPOCH: 8
Loss=0.02687823586165905 Batch_id=937 Accuracy=98.65: 100%|█| 938/938 [00:34<00:

Test set: Average loss: 0.0405, Accuracy: 9877/10000 (98.77%)

EPOCH: 9
Loss=0.024293921887874603 Batch_id=937 Accuracy=98.78: 100%|█| 938/938 [00:33<00

Test set: Average loss: 0.0379, Accuracy: 9885/10000 (98.85%)

EPOCH: 10
Loss=0.10666413605213165 Batch_id=937 Accuracy=98.98: 100%|█| 938/938 [00:34<00:

Test set: Average loss: 0.0369, Accuracy: 9876/10000 (98.76%)

EPOCH: 11
Loss=0.02819644846022129 Batch_id=937 Accuracy=99.19: 100%|█| 938/938 [00:34<00:

Test set: Average loss: 0.0288, Accuracy: 9904/10000 (99.04%)

EPOCH: 12
Loss=0.0033064689487218857 Batch_id=937 Accuracy=99.36: 100%|█| 938/938 [00:35<0

Test set: Average loss: 0.0253, Accuracy: 9908/10000 (99.08%)

EPOCH: 13
Loss=0.022911909967660904 Batch_id=937 Accuracy=99.45: 100%|█| 938/938 [00:34<00

Test set: Average loss: 0.0269, Accuracy: 9908/10000 (99.08%)

EPOCH: 14
Loss=0.010325931943953037 Batch_id=937 Accuracy=99.55: 100%|█| 938/938 [00:34<00

Test set: Average loss: 0.0252, Accuracy: 9909/10000 (99.09%)
