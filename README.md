# Deep Learning-Based Encrypted Traffic Classification: Novel Use of the Headers
### 



* This is a Pytorch implementation of 1D-CNN Deep Packet.

* See ```107research_poster.pdf``` for details.

## Dataset

* This research uses the [***UNB ISCX VPN-nonVPN dataset***](https://www.unb.ca/cic/datasets/vpn.html).

* Make sure to download and unzip the dataset. Then run the following command for data preprocess:
    ```
    python3 dataloader.py {path/to/unzipped/dataset}
    ```
* The preprocessed dataset is saved to ```./pre```.

## Run

* Make sure you have run the previous steps so that the preprocessed dataset is in ```./pre``` folder.

* To train the model and see the results, simply run:
    ```
    python3 main.py
    ```
