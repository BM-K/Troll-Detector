# Troll Detector (악플 분류기)
- Model
  - SKT-Brain KoBERT
    - https://github.com/SKTBrain/KoBERT
- Data
  - 한국어 악성댓글 데이터셋
    - https://github.com/ZIZUN/korean-malicious-comments-dataset

## Installation
```
git clone https://github.com/BM-K/Troll-Detector.git
cd Troll_Detector
git clone https://github.com/SKTBrain/KoBERT.git
cd KoBERT
pip install -r requirements.txt
pip install .
```

## Training Models & hyperparameters
```
python train_main.py \
  --drop_rate 0.1 \
  --max_len 64 \
  --batch_size 256 \
  --epochs 10 \
  --patience 5 \
  --lr 0.0001 \
  --warmup_step 0.2 \
  --train_ True \
  --test_ True \
  --inference True \
  --path_to_data ./data \
  --path_to_sorted ./output \
```

하이퍼 파라미터는 위와 같이 튜닝 하였고, **88.2% Accuracy**를 얻었습니다. <br>
학습이 완료 되어 inference만 수행 시 train_, test_ 를 False로 inference 는 True로 설정해 사용할 수 있습니다.

## Data Split
|Train|Dev|Test|
|------|------|------|
|9000|500|500|

## Demo
```
> 아무리 사람들이 당신을 욕해도 난 당신 편이예요 홧팅!! [Normal]
> 아무리 사람들이 당신을 커버쳐도 난 너 같은 쓰레기 취급 안 한다~ [Troll]

> 아오 저기서 저렇게 휘두르냐... 아쉽다 [Normal]
> 아오 저기서 저렇게 휘두르면 안 되지 쉬더니 퇴물 됐네 [Troll]

> 확실히 영화 하나는 잘 만드네 근데 배우 캐스팅은 별로 ㅋㅋ; [Normal]
> 확실히 영화 하나는 잘 만드네 근데 배우는 개못생김 ㅋㅋ;	[Troll]
```

<img src = 'https://user-images.githubusercontent.com/55969260/99531889-c472b200-29e6-11eb-83e5-f657d91b6224.gif'>
