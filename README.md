# MBTI Personality Classification based on users' social media posts
------
![steps](https://github.com/Pranshu-Bahadur/nlp-mbti/blob/RSJdoc/final.gif)
<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project  
   Myers Briggs Type Indicator personality Classification based on users' from social media posts.

### Built With

* [Python 3.6]()
* [Pytorch 1.8]()

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Dependencies
------

* pip
  ```sh
   pip install torch pandas transformers tokenizers datasets numpy
  ```
# Usage
---

## Command line Arguments

Here the user has to input the required command line arguments to run the model.

It is recommended for an intial user to run "python main.py --help" , for information on each argument.

```sh
"-m" -> Model name.

"-d" -> Dataset directory path.

"-dl" -> Next delimiter.

"-w" -> Minimum words per post.

"-tb" -> Training batch size.

"-eb" -> Eval batch size.

"-r" -> Test split ratio.

"-l" -> Learning rate.

"-wd" -> Weight decay.

"-n" -> Num classes.

"-ml"-> Multilabel classification.

"-f" -> Number of epochs.

"-mt"-> Metrics.

"--optimizer" -> Optimizer.

"--loss" -> Loss criterion.

"--train"->To train model.

"-o"-> Output directory .

"--save_interval" 
```
## Example Cli Command
------

```sh
python main.py -m vinai/bertweet-base -d ./mbti_1.csv -dl "|||" -w 5 -tb 256 -eb 256 -r 0.75 -l 176e-06 -wd 1e-05 -n 4 -f 5 --optimizer ADAM --loss BCE --train -o ./ops --save_interval 2
```
