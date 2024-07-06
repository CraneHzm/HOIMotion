# HOIMotion: Forecasting Human Motion During Human-Object Interactions Using Egocentric 3D Object Bounding Boxes
Project homepage: https://zhiminghu.net/hu24_hoimotion.


## Abstract
```
We present HOIMotion – a novel approach for human motion forecasting during human-object interactions that integrates information about past body poses and egocentric 3D object bounding boxes. 
Human motion forecasting is important in many augmented reality applications but most existing methods have only used past body poses to predict future motion. 
HOIMotion first uses an encoder-residual graph convolutional network (GCN) and multi-layer perceptrons to extract features from body poses and egocentric 3D object bounding boxes, respectively. 
Our method then fuses pose and object features into a novel pose-object graph and uses a residual-decoder GCN to forecast future body motion. 
We extensively evaluate our method on the Aria digital twin (ADT) and MoGaze datasets and show that HOIMotion consistently outperforms state-of-the-art methods by a large margin of up to 8.7% on ADT and 7.2% on MoGaze in terms of mean per joint position error. 
Complementing these evaluations, we report a human study (N=20) that shows that the improvements achieved by our method result in forecasted poses being perceived as both more precise and more realistic than those of existing methods. 
Taken together, these results reveal the significant information content available in egocentric 3D object bounding boxes for human motion forecasting and the effectiveness of our method in exploiting this information.
```


## Environments:
Ubuntu 22.04
python 3.8+
pytorch 1.8.1
cudatoolkit 11.1


## Usage:
Step 1: Create the environment
```
conda env create -f ./environments/hoimotion.yaml -n hoimotion
conda activate hoimotion
```


Step 2: Follow the instructions in './adt_processing/' and './mogaze_processing/' to process the datasets.


Step 3: Set 'data_dir' and 'cuda_idx' in 'train_mogaze_xx.sh' (xx for p1, p2, p4, p5, p6, or p7) to evaluate on different participants. By default, 'train_mogaze_xx.sh' first trains the model from scratch and then tests on different actions. If you only want to evaluate the pre-trained models, please comment the training commands (the commands without the 'is_eval' setting).


Step 4: Set 'data_dir' and 'cuda_idx' in 'train_adt.sh' to evaluate. By default, 'train_adt.sh' first trains the model from scratch and then tests on different actions. If you only want to evaluate the pre-trained models, please comment the training commands (the commands without the 'is_eval' setting).


## Citation

```bibtex
@article{hu24_hoimotion,
	author={Hu, Zhiming and Yin, Zheming and Haeufle, Daniel and Schmitt, Syn and Bulling, Andreas},
	journal={IEEE Transactions on Visualization and Computer Graphics}, 
	title={HOIMotion: Forecasting Human Motion During Human-Object Interactions Using Egocentric 3D Object Bounding Boxes}, 
	year={2024}}
```