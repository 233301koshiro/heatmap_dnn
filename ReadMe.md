#　gnnの使い方
```
python new_debug.py --merge-dir ./merge_joint_robots --epochs 500 --batch-size 16 --loss-weight '1.0 ,5.0 ,1.0 ,0.25 ,0.25 ,0.25 ,0.25 ,3.33 ,3.33 ,3.33 ,3.33 ,3.33 ,3.33' --mask-mode none --mask-k 0 --seed 42 --save-dir ./checkpoints --log-csv ./checkpoints/training_log.csv --metrics-csv ./checkpoints/test_metrics.csv
```
loss_weightに関して
元はoriginとaxisだけ10のほかすべて0.1だったが，joint_typeやaxis,originなどは複数の特徴量から一つのlossをだしてまたそのlossを複数に転記するため
単純な話lossの影響が特徴量数に比例する．
つまり4つからなるjoint_typeは1/4の0.025に,3つからなるaxisは1/3の3.3にすることで全体のlossに与える影響を均一化している．

# headmap dnn

## 準備
### docker imageの準備
```
cd docker 
./build.sh
```

## datasetの準備
- 入力画像とラベル画像のペアを用意し，その対応をcsv形式で記載する．
- 入力画像とラベル画像は同じ縦横サイズの画像とする．
    - 入力画像はカラー画像
    - ラベル画像は0と255のみで構成されるグレースケール画像
    - 試した解像度は256x256のみだが他の解像度でも問題ないはず．
- csvファイルに記載する画像ファイルのパスはcavファイルからの相対パス，もしくは，絶対パスで記載する．
    - csvの例
        ```
        train/image_000000.png,train/label_000000.png 
        train/image_000001.png,train/label_000001.png 
        train/image_000002.png,train/label_000002.png 
        train/image_000003.png,train/label_000003.png 
        ```
- training用のセットとvalidation用のセットを作成する．

### (参考) ms coco datasetを用いたデータセットの作成
- ms cocoのダウンロード
    ```
    cd ..
    mkdir -p dataset/coco
    cd dataset/coco
    wget http://images.cocodataset.org/zips/train2017.zip
    wget http://images.cocodataset.org/zips/val2017.zip
    wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
    unzip train2017.zip
    unzip val2017.zip
    unzip annotations_trainval2017.zip
    ```
- docker環境の起動
    ```
    cd docker
    ./run.sh
    ```
- 生成コードの実行
    ```
    python generate_mscoco_dataset.py
    ```

## 学習
- docker環境の起動
    ```
    cd docker
    ./run.sh
    ```
- 学習スクリプトの実行(例)
    ```
    python train.py --traincsv train.csv --valcsv val.csv
    ```

- 学習状況の確認(別ターミナルで確認)
    ```
    cd docker 
    ./exec
    ```
    ```
    tensorboard --logdir . --bind_all
    ```
    ブラウザで http://localhost:6006 へアクセスすると，training loss等のグラフを見ることができる．
### 学習スクリプトのオプション
- `--traincsv` : トレーニング用データを記述したcsvファイル
- `--valcsv` : バリデーション用データを記述したcsvファイル
- `--batchsize` : ミニバッチサイズ (default : 64)
- `--epoch` : 学習エポック数 (default : 10)
- `--lr` : 学習率 (default : 5e-4)
- `--output` : 出力ディレクトリ
    - 指定がない場合は"output-yyyyMMddhhmmss"を作成する．
    - 指定したディレクトリが既にある場合は，プログラムが止まるので注意が櫃お湯
- `--device` : 学習を行うデバイス(cpu or cuda)を指定する (default : cuda)
- `--inputsize` : 学習時に入力する画像サイズ
    - 指定しない場合は画像サイズと同じ (default)
    - 1つ指定した場合は1辺が指定したサイズの正方形とする
    - 2つ指定した場合は(横，縦)のサイズの矩形とする．
    - 3つ以上指定した場合は先頭2つの数を(横，縦)のサイズの矩形とし，それ以降は無視する．
- `--useamp` : AMPを使うかを指定する．(default : false)
- `--arch` : ネットワークのアーキテクチャを指定する．指定できる種類については[こちら](https://smp.readthedocs.io/en/latest/models.html)を参照．(default : Unet)
- `--encoder` : ネットワークのエンコーダを指定する．指定できる種類については[こちら](https://smp.readthedocs.io/en/latest/encoders.html)を参照．(default : resnet18)
- `--classnum` : 出力クラス数を指定する．(default : 1)

## 推論
- docker環境の起動
    ```
    cd docker
    ./run.sh
    ```
- 推論スクリプトの実行(例)
    ```
    python3 pred.py --input val/image_00000000.png --label val/label_00000000.png --output result_00000000.png 
    ```

## 参考webページ
- pytorch
    - Segmenation ModelのgithubとDocumentation
        - https://github.com/qubvel-org/segmentation_models.pytorch
        - https://smp.readthedocs.io/en/latest/index.html
    - Segmentation Modelの使い方
        - https://qiita.com/tchih11/items/6e143dc639e3454cf577
            - 情報がちょっと古い
        - https://zenn.dev/takiser/articles/35f33b7405a29b
    - pytorchのDataset, Dataloaderの作り方
        - https://qiita.com/mathlive/items/2a512831878b8018db02
    - pytorchの学習ループ
        - https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
- AMP (Automatic Mixed Precision examples)
    - これは近年のGPU(RTX2000番以降)の性能を生かすための手法．半精度浮動小数(FP16)を用いるので，精度は悪くなるが，使用メモリ量が半減するため，バッチサイズを大きくとれる．
        - https://tawara.hatenablog.com/entry/2021/05/31/220936
        - https://qiita.com/bowdbeg/items/71c62cf8ef891d164ecd
- OpenCV
    - OpenCVグレースケールからカラーマップへの変換
        - https://qiita.com/hakoyam/items/e312af5c3b9c9ae58fff
