# Audio-Speech-classification-models

본 레포는 음성 분류 문제에서 사용되는 모델들을 담고있다. 

---

음성 분야는 다른 분야에 비해 데이터가 현저히 부족하다. 학습을 시키는데 충분한 양이 보장되어야하는 딥러닝 모델의 특성상 이는 치명적이다. 이 때문에 초기의 음성 처리 분야에서는 전이 학습이 활발하게 이루어졌다.

![0\*6isarY9jjYNmNvtA](https://miro.medium.com/max/1400/0\*6isarY9jjYNmNvtA)이처럼, 음성 피쳐들 즉, signal representation 값들을 사전 학습된 모델에 태워 feature extraction을 하게 된다. 이때 일반적으로 이미지 모델들이 사용되는데, 이는 스펙트로그램, 스펙트럼 등을 하나의 신호로 보는 것이 아니라 그것을 **시각화해서 마치 이미지처럼 여길 수 있기 때문이다.**

하지만, 최근에 나타난 sota 모델들은 전이학습을 사용하지 않고 오히려 augmentation technique에 집중해서 음성 데이터의 수를 늘리는 것에 집중한 것이 대다수이다.

## 2-2. Network Architectures

### 2-2-1. Image Pretrained Models

#### 1) AlexNet <span dir="">(2012)</span>

![](https://lh4.googleusercontent.com/mIZVZLRo-rij0rqPVyZ4W2OGbox4rSVLv-7wygMwTa9VbN-V8GFGtpBQczEk6smwLNiSL-x3_dVYhiE08x1PHMgbc7ETAOk0ibM2TocO70nu0hGYYlV-LI48nKqtC02weuRPWa9ogF-r)

<span dir="">딥러닝의 황금기를 시작하게 한 모델로, CNN이 세간의 주목을</span> 받게된 계기가 되었다. 구성은 다음과 같다:

* <span dir="">conv layer, max-pooling layer, dropout layer 5개</span>
* <span dir="">fully connected layer 3개</span>
* <span dir="">nonlinearity function : ReLU</span>
* <span dir="">batch stochastic gradient descent</span>

<span dir="">AlexNet이 중요한 이유는 의미있는 성능을 낸 첫번째 CNN 아키텍처이자, AlexNet에 쓰인 드롭아웃 등 기법은 이 분야 표준으로 자리잡을 정도로 선도적인 역할을 했기 때문이다.</span>

#### 2) VGG <span dir="">(2014)</span>

<span dir="">AlexNet 이후 층을 더 깊게 쌓아 성능을 높이려는 시도들이 계속되었는데, VGG는 그 대표적인 모델이다. vgg의 구조는 다음과 같</span>다.

![](https://lh3.googleusercontent.com/cmmiOT68DsW-iKd6vz5GKk_Osv8KGgr-c3ev50OdSaQx_lH_G82zhZRdY2hU2oOTmOIqAAMV4n1ym5n2jBB-zPtKFDSqEVeI-BNk2pOLDrcebvmqK5G6z0OB6bJKWbxdoz2niyWW1Vap)

<span dir="">보다시피, conv layer가 5개였던 alexnet과 달리, 16개의 conv layer로 구성되었다는 것을 볼 수 있다. </span>

#### 3) Inception/GoogLeNet <span dir="">(2014)</span>

VGG와 마찬가지로 AlexNet의 구조를 더 딥하게 만들기 위한 모델 중 하나이다. 대신, 그냥 모델을 쌓는 것이 아니라, 한 레이어 자체를 두텁게 만드는 것을 시도한다.

<span dir="">Google 연구원들은 한 가지의 conv filter를 적용한 conv layer를 단순히 깊게 쌓는 방법도 있지만, 하나의 layer에서도 다양한 종류의 filter나 pooling을 도입함으로써 개별 layer를 두텁게 확장시킬 수 있다는 아이디어를 제시한다. 이들이 제안한 구조가 바로 **Inception module**</span>이다. (하단)

![](https://lh6.googleusercontent.com/Pfk2bnWA3aCDaOMFuwWFH5u6fLsGXdILZG330bkl48GQXGaXWdRfdhZ38Ro8AK6Mk7Y8n_wMo1DG4RX1LhyhTouXR5Fhx2L7a3GwEKxnY0UM_M6BWtoXbzuHIjjbeari1k8COVH7Dvcj)

<span dir="">Inception은 다른 Conv Layer와 다르게 하나의 Input에 대해 여러 종류의 크기를 가진 Filter를 병렬로 적용한 뒤 하나로 모아 출력한다. Inception module에서 특히 주목받은 것이 바로 1×1 conv filter인데, 해당 필터는 차원을 줄이면서 fully connected layer와 유사한 기능을 수행할 수 있어 주목 받았다.이 Inception 구조를 이용해서 만든 모델이 GoogLeNet이다.</span>

![](https://lh5.googleusercontent.com/IMCxhPO43_L8uT1T1KD7uJv__7og0zB2FHeNr11gGTRXvkK-oMzJNfro-eEWUFa336aujI0qhUwB9jyHNr9RxrFxFQWCTzXmQ8Ir3hTXzOsoZuvu4w2A42BCNO-g1Ff3DZScp8GzdEhM)

#### 4) ResNet<span dir=""> (2015)</span>

<span dir="">AlexNet이 처음 제안된 이후로 CNN 아키텍처의 층은 점점 더 깊어졌다. AlexNet이 불과 5개 층에 불과한 반면 VGGNet은 16개 층, GoogLeNet은 22개 층에 달한다. 하지만 층이 깊어질 수록 기울기 소실 문제가 심해</span>졌다. <span dir="">ResNet 저자들의 **residual block**을 도입해 기울기 소실 문제를 완화시켰다. (하단)</span>

![](https://lh5.googleusercontent.com/4glO54-FenB9sPhTxbVvJ1OAmvmwPnzkUtX9Nn9eSmCR3k9aMy8OiIX8LCAgNgO-4ma8nu4dUkzKN9fv_NNpfnFapLRFAnNseFUeXvPdOKTAxxdxQRa_XYpHSE3UYQn_aAggZs6YF6X2 "source: imgur.com")

<span dir="">즉,gradient가 잘 흐를 수 있도록 일종의 지름길(shortcut, skip connection)을 만들어 </span>준 것이다. <span dir="">최종적으로 하단의 모델이 만들어진다. </span>

![](https://lh3.googleusercontent.com/VUf4MLVT6KqgeKSkUHwsuEtM7nWapndRyajlENkUs-75Dwayr0-xstRx_JJPWcdXsdQ4B9M_K8VcWBiIeyTPFIpERSNx7r30_uZq7_wtmdvPnc4ffBYKLCzXc66_Q_voKrF5u20Wvy2L)

#### 5) DenseNet<span dir=""> (2016)</span>

<span dir="">ResNet이 residual의 개념을 도입한 것이 성능을 크게 끌어올리자, </span>[<span dir="">DenseNet</span>](https://arxiv.org/pdf/1608.06993.pdf)<span dir="">(2016)은 ResNet에서 한발 더 나아가 전체 네트워크의 모든 층과 통하는 지름길을 만들었다. 이는 conv-ReLU-conv 사이만 뛰어넘는 지름길을 만들었던 ResNet보다 훨씬 과감한 시도이다.</span> 모델 구조는 하단과 같다:

![](https://lh3.googleusercontent.com/_sEE1c6R__dlGN4Lj9sOVeigmbEbVZ8G229UY3I2VETOihHclJNMivZ9MxOmP30jj4btfg6QqqDmUNSkRqUW9mmbfP2OgEDnGzs4kEcTUVLCapjfVpHkZHGSNenOgxx3sGIOskUhovj9 "source: imgur.com")

#### 6) SqueezeNet<span dir=""> (2016)</span>

**<span dir="">AlexNet의 파라미터 수를 낮추고 정확도는 유지시키는 방법에 집중한 모델로,</span>**<span dir=""> </span>모델의 파라미터를 크게 감소시킬 수 있는 Fire module을 바탕으로 모델을 구성한다.

![](https://lh4.googleusercontent.com/jOODqWx2q-b-OB4WrZJlKntZFIzykBpSH7WDnt08fFhVJMLMg-M8-W24eIKgS-uU8iOgcorVVi66Gzhed7biE0lL0DhDvy7bZ20gd3ROW8DgYquux9HfwPmjczaEJ0Qu7lZKa4oeZtLV)<span dir="">주요 전략은 다음과 같다:</span>

1. <span dir="">3x3 filter를 1x1 filter로 대체 </span>
2. <span dir="">3x3 filter로 입력되는 입력 채널의 수를 감소 </span>
3. <span dir="">pooling layer를 최대한 늦게 배치하여 만들어짐</span>

<span dir="">1과 2를 활용하여 parameter 수가 줄어드는데, 이것들이 fire module에 들어간다. 자세히 살펴보면 다음과 같</span>다.

![](https://lh6.googleusercontent.com/lqJpmp1Vs96tUUHBRZHJ4_TNP_75M3d_V38n2u6X25bnMbabpdFcynF0AtyPZeAnBrhjIH0iy3XzhF2DgCITrGvswsuUF5lCUxR2wgk3AcIf3uNISABECaUkxsyUgnORyaY3TpfCJbvk)

### 2-2-2. Audio Models

#### 1) SincNet <span dir="">(2018)</span>

<div>

싱크넷(sincnet)은 단순하게 **컨볼루션 필터가 싱크 함수인 1D conv**이다. SincNet을 이용하면 보다 빠르고 가벼우면서도 정확하게 음성의 피쳐를 추출할 수 있다. 구조를 살펴보면 다음과 같다.

![enter image description here](https://i.imgur.com/n1EXsWV.png)

위의 레이어들은 사실 다른 모델들과 크게 다른 점이 없다. 발화자가 누구인지 맞추는 과정을 통해서 가중치들을 수정/학습해나간다 정도만 이해하면 된다. 중요한 것은 첫 번째의 sinc 함수가 적용된 conv filter이다. 각 필터들은 원시 음성에서 우리의 태스크인, 발화자가 누구인지 맞추기 위해 중요하다고 판단되는 주파수 영역대의 정보를 추출해 상위 레이어로 보낸다.

즉, 우리가 SincNet을 이해하기 위해서는 1) time-domain에서 convolution 연산이 어떻게 진행되는지 2) sinc 함수가 무엇인지를 알아야 한다.

* time-domain에서 convolution 연산

시간 도메인에서의 컨볼루션 연산의 정의는 다음과 같다.

`y[n] = x[n]*h[n] for n in range(0,L)`

x\[n\] 은 시간 도메인에서의 n 번째 raw wave sample, h\[n\] 은 컨볼루션 필터(filter, 1D 벡터)의 n 번째 요소값, y\[n\] 은 컨볼루션 수행 결과의 n 번째 값, L 은 필터의 길이(length)를 나타낸다. 예를 들어서 y\[1\]에 대한 수행 결과는 다음과 같다.

`y[1]=x[0]⋅h[1]+x[1]⋅h[0]`

![enter image description here](https://i.imgur.com/x2qYYXw.jpg)

즉, 컨볼루션 연산 결과물인 y 는 입력 시그널 x 와 그에 곱해진 컨볼루션 필터 h 와의 관련성이 높을 수록 커진다. 컨볼루션의 형태에 따라서 어떤 입력값을 완전히 없애버릴수도 있다. 다시 말해 **컨볼루션 필터는 특정 주파수 성분을 입력 신호에서 부각하거나 감쇄**시킨다는 것.

그런데, 여기서 우리가 집중해야할 것은 **시간(time) 도메인에서의 convolution 연산을 주파수(frequency) 도메인에서의 곱셈 연산과 동일**하게 취급할 수 있다는 것이다.

이게 무슨 말인가 보면, ![enter image description here](https://i.imgur.com/w3ODRrt.jpg)

time domain의 가운데가 conv라고 가정했을 때, 저런 형태의 conv filter는 입력값을 그대로 반환하게 될 것이다. 그런데, 이와 유사하게 freq 영역에서도 같은 값을 반환하게끔 하는 곱셈 연산이 존재.

![enter image description here](https://i.imgur.com/bVDA6Qo.jpg)

다음 예시를 보면, 밑의 그림과 같이 직사각형의 형태를 가지는 함수(구형 함수)를 주파수 도메인에서 입력 신호와 곱하면, f1 과 f2 사이의 주파수만 남고 나머지는 없어질 것이다. 이때 우리는 이에 시간 도메인에서 컨볼루션 연산을 수행한 결과와 대응시킬 수 았다. 그리고 이 때 컨볼루션에서 사용하는 함수가 싱크 함수(sinc function)이다.

* Sinc function

태스크를 더 정확하게 수행하기 위해서는 Bandpass filter를 거쳐야 한다. 이는 앞서 언급했듯이 특정 주파수 영역대만 남기는 역할을 하는 함수이다. 그리고 당연하게도, 특정 주파수 영역대를 뽑아내는 것이기에 가장 이상적인 필터의 모양은 직사각형이다.

![enter image description here](https://i.imgur.com/FgzqVBY.jpg)

이때 우리는 주파수 도메인과 시간 도메인을 매핑해줄 수 있다. 바로 싱크 함수(Sinc function)를 통해서다. **주파수(frequency) 도메인에서 구형 함수(Rectangular function)으로 곱셈 연산을 수행한 결과는 시간(time) 도메인에서 싱크 함수로 컨볼루션 연산을 적용한 것과 동치이다.**

![rec](https://i.imgur.com/u1xY7P1.png)

구형 함수는 위와 같고, 싱크 함수는 sin(x)를 x로 나눈 것이다. `sinc (x)=sin(x)/x`. 둘은 서로 변환이 가능한데, 싱크 함수를 푸리에 변환(Fourier Transform)한 결과는 구형 함수가 되며, 이 구형 함수를 역푸리에 변환(Inverse Fourier Transform)하면 다시 싱크 함수가 된다. 즉, 두 함수는 푸리에 변환을 매개로 한 쌍을 이루고 있다는 이야기.

![enter image description here](https://i.imgur.com/2cZp0Ky.png)

그 역도 성립한다. 구형 함수를 푸리에 변환한 결과는 싱크 함수가 된다. 이 싱크 함수를 역푸리에 변환을 하면 다시 구형 함수가 된다.

![enter image description here](https://i.imgur.com/hCoYfkY.png)

그런데, 문제는 싱크 함수를 통해 완전한 구형 함수를 얻어내려면 싱크 함수의 길이 $L$이 무한해야 한다는 것이다. 이는 현실적으로 가능하지 않으므로 싱크함수를 유한한 길이로 자르는 과정이 포함된다. 길이별 비교는 다음과 같다.

![enter image description here](https://i.imgur.com/dPiXPQ6.png)

보면 알겠지만 우리가 바라던 이상적인 모양인 사각형과 점점 달라지기에 원하는 주파수 정보는 덜 얻게 되고 필요없는 정보들이 자꾸 추가된다. 이 때문에, 단순히 특정 길이로 자는 것이 아니라 hamming windowing을 한다. 이는 필터의 양끝을 스무딩한다는 것이다.

![PNG 이미지3](https://i.imgur.com/QGo2o7U.png)

sinc 함수의 해밍 윈도우를 푸리에 변환한 결과이다. 결과를 보면 알 수 있듯, 중심 주파수 영역대는 잘 캐치하고 그 외 주파수 영역대는 무시한다.

![enter image description here](https://i.imgur.com/Hsi7qpn.png)

* SincNet Filter

이제 SincNet의 첫 번째 레이어를 살펴보겠다. SincNet은 푸리에 변환(Fourier Transform) 없이 시간 도메인의 입력 신호를 바로 처리한다. 시간 도메인과 같에 따른 컨볼루션은 다음과 같았다.

`y[n] = x[n]*h[n].`

이곳에 우리는 우리가 정의한 싱크 함수(컨볼루션 필터)를 적용한다.

`y[n]=x[n]∗g_w[n,f1,f2]`

`g_w[n;f_1,f_2]=(2f_2sinc(2πf_2n) − 2f_1sinc(2πf_1n)) * w[n]`

f_1과 f_2는 bandpass 범위에 해당하는 스칼라 값으로, f/2f_1에서 f/2f_2 사이만 남기고 나머지 주파수 영역대는 무시한다. 첫 번째 괄호 안의 식은 해당 영역의 구형 함수를 시간 도메인으로 옮긴 결과이다. 이때, f_1과 f_2는 learnable parameter이며, 저자들은 전자를 low cut-off frequency, 후자를 high cut-off frequency라고 부른다. 마지막의 w\[n\]은 hamming window이다.

</div>

#### 2) Wav2Vec 2.0

End2End 방식으로 MLM 학습 방식과 CPC training을 함께 적용할 수 있는 모델이다. 

![wav2vec2](uploads/f4a8bc70629a6da7577cfef6f6e55113/wav2vec2.png)

Wav2Vec2.0의 구조는 위와 같다. 

크게 2 부분으로 이루어지는데, 하단에서 일정 길이의 음성을 CNN을 통해 특징 벡터(latent speech representaion)으로 변환하는 부분이 **Encoder**, 이를 공유 벡터(Context representation)으로 변화라는 부분이 **Aggregator**이다. wav2vec2.0에서는 transformer 모델을 사용한다. 

Wav2vec2.0의 최종적인 목적은 CPC이다. CPC란 Contrastrive Predictive Coding의 준말로, **음성 데이터만으로 좋은 representation vector를 추출하는 모델 개발하는 것이 목적**이다. 이때, 좋은 representation이란 high-level, 즉 음성 전반적으로 공유되는 정보, noise로 인해 잘 변하지 않는 정보를 의미한다. CPC는 Context Representation(공유벡터)를 활용하여 일정 거리에 위치한 Latent speech representation(특징벡터)를 예측하게 모델링하여 모델이 공유벡터의 공유정보를 추출하게 하고 거리와 관련된 패턴을 학습한다. 거리 벡터는 거리에 따른 정보의 변화를 학습하기 때문에, 과거의 특징 벡터에 공통적으로 들어있는 정보로 미래 특징 벡터를 예측할 수 있게 한다. 

추가적으로 알아야 할 부분은 Quantization이다. wav2vec2.0은 BERT의 **MLM 방식을 이용해서 학습을 진행**한다. 이를 위해서는 토큰화가 수행되어야하는데, 연속적인 음성 신호를 이산화하는 과정이 vector quantization이다. 

#### 3) Audio Spectrogram Transformers<span dir=""> (2021)</span>

![](https://lh5.googleusercontent.com/fpG-JImePKEr9edTod3F3m5tzdixPqr2w8MdosO5i_roh6YZtPadgiXVi8BbgwQ6aXhEix06sZAQwxZF5TIgldVrX1T18sKK9_eF3T0hw_jnikRpcGLExLEhHdSgcBoQe2z57dwvIuzN)

<span dir="">Convolution을 사용하지 않고 attention을 사용한 첫 번째 모델</span>로, Transformers의 Encoder만을 떼어와서 audio의 Mel Spectrogram 이미지로 pretraining 시킨 모델이다. <span dir="">Audio classification에 최적화된 모델</span>이다. 

다음의 [링크](https://github.com/YuanGongND/ast)를 통해 들어가면 저자들이 공개한 코드와 pretrained model을 다운받을 수 있다. 
