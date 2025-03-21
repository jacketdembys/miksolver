# `Paper`: "LBE-DDIK: Is One Model Good Enough to Learn-by-Example the Inverse Kinematics of Multiple Serial Robots?"


## <div align="center">Requirements</div>
```shell
- pytorch:2.0.1
- cuda11.7
- cudnn8
- wandb
- scikit_learn
- numpy
- scipy
- pandas
- matplotlib
- tqdm
```


## <div align="center">Usage</div>

</details>
<details open><summary>Clone repository</summary>

```shell
git clone https://github.com/jacketdembys/miksolver.git
cd miksolver
```

</details>



</details>
<details open><summary>Generate datasets </summary>
Choose the dataset type in the generate_dataset.py file and run it to generate the corresponding dataset:

```shell
python generate_dataset.py
```

</details>

</details>
<details open><summary>Train IK model</summary>
Choose/set the training configurations in the create_experiments.py file, then create a train.yaml configuration file:

```shell
python create_experiments.py
```

Run the training script to train/eval/test the model:

```shell
python mik-solver.py --config-path train.yaml
```

<!---
</details>

</details>
<details open><summary>Test IK model (ToDo)</summary>
</details>
-->