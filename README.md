# MBTI Personality Classification based on users' twitter posts
------
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
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
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

### Prerequisites

This is  list of python packages you need to use the software and how to install them.


* pip
  ```sh
   pip install torch 
   pip install pandas 
   pip install transformers 
   pip install tokenizers 
   pip install datasets 
   pip install numpy
  ```

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Pranshu-Bahadur/nlp-mbti.git
   ```
2. Install pip packages mentioned above

# Command line Arguments

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
# Example

python main.py -m vinai/bertweet-base -d ./mbti_1.csv -dl "|||" -w 5 -tb 256 -eb 256 -r 0.75 -l 176e-06 -wd 1e-05 -n 16 -f 5 --optimizer ADAM --loss BCE --train -o ./ops --save_interval 2

# Steps



## Features 

- [x] (unit tested) utils.py : dataframe parser using reduce & singledispatch
- [ ] agent.py : NLPAgent(kwargs) | Just build constructor & tokenizer based on user kwargs




## Preprocessing strategy:
- [x] 50 posts per user, split by "|||" delimiter.
- [ ] BertweetTokenizer : normalization=True, fast=False (Which will normalize tweets (add URL tokens) and use then bpe) | Padding according to max words after split.
