# Home office ergonomics

If you sit behind a desk for hours at a time, you're not doomed to a career of neck and back pain or sore wrists and fingers. Proper office ergonomics — including correct chair height, adequate equipment spacing and good desk posture — can help you and your joints stay comfortable at work.

## How does this work
Machine learning packages available for Python


- [Scikit-learn](https://scikit-learn.org) - open source Python machine learning library supporting supervised learning & weightings
- [Tensorflow Keras](https://www.tensorflow.org/guide/keras/overview) - high-level API to build and train models 

### Capturing photos of "good posture" and "slumped posture" 
We need to train supervised learning

## Python 3

```
which python3


virtualenv -p `which python3` venv
source venv/bin/activate
python --version
pip --version
pip install -r requirements.txt 
```


## Capture

Press the "space" bar to stop capture. Note: the first run may take a few minutes to start

```
python posture-watch.py --capture-good
python posture-watch.py --capture-slump
```

## Train

```
python posture-watch.py --train
```


## Live Video

```
python posture-watch.py --live
```

## Cleanup

```
rm -fr train/
```

