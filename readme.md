<h1 align="center">
  <b>fever-transformers</b>
</h1>

<p align="center">
  📄 Evidence Retrieval and Claim Verification for the FEVER shared task using Transformer Networks
  <br/>

  <sub>
    Available pre-trained models: Bert, RoBERTa, Albert, XLM-RoBERTa
  </sub>
</p>

# FEVER Shared Task
The [FEVER Shared Task][link:fever] is a task in which participants are asked to classify a `claim` (e.g. "The number 42 is the Answer to the Ultimate Question of Life, the Universe, and Everything") into `SUPPORTS`, `REFUTES` or `NOT ENOUGH INFORMATION` while also providing the relevant evidence sentences from Wikipedia (only).

## Run the pipeline

The project is provided with a CLI that allows you to run the pipeline on your machine with ease.

If you want to run it locally, you need to run the following commands.

```bash
# Install pipenv (i.e. a better python package manager).
pip3 install pipenv

# Download the source code from GitHub and cd into the folder.
git clone --branch master https://github.com/simonepri/fever-transformers
cd fever-transformers

# Run the entire pipeline (it also installs the required dependencies).
bash scripts/pipeline.sh
```

> NB: The only requirement to run the mentioned commands, is that you have python3 and pip3 installed on your machine. You can also skip the pipenv installation if you already have it on your machine.

Alternatively, the CLI allows you to run the single tasks individually as follows.

```bash
# Install required dependencies in a virtual environment.
bash scripts/pipeline.sh install_deps

# Download the fever shared task.
bash scripts/pipeline.sh download_fever

# Build an sql database using the fever wikipedia dump.
bash scripts/pipeline.sh build_db

# Process the datasets through the UKP-Athene document retrieval model.
# Alternatively you can run bash scripts/release.sh --download output ukp-athene
bash scripts/pipeline.sh document_retrieval

# Process the datasets through the transformer network sentence retrieval model.
# See below for the possible values of model type and name.
bash scripts/pipeline.sh sentence_retrieval --model-type bert --model-name bert-base-cased

# Process the datasets through the transformer claim verification model.
# See below for the possible values of model type and name.
bash scripts/pipeline.sh claim_verification --model-type bert --model-name bert-base-cased

# Combine the results from the previous steps to generate the final submission files.
bash scripts/pipeline.sh generate_submission --force
```

> NB: If you run a task multiple times, the CLI will execute actions for that task on an as-needed basis (e.g. if the finetuned model is already available it wont start the finetuning process again).

The following flags can be used to modify the behavior of the CLI.

| Flag | Purpose |
|------|---------|
| `‑‑force` | Delete the folder containing the data of the task and then start the task |
|  `‑‑model‑type` | Set the transformer model to use. It can be one of: <br/>`bert`, `xlnet`, `xlm`, `roberta`, `distilbert`, `albert`, `xlmroberta` |
| `‑‑model‑name` | Set the pretrained checkpoint of the model to use. It can be one of: <br/> `bert-base-uncased`, `bert-large-uncased`, `bert-base-cased`, `bert-large-cased`, `bert-base-multilingual-uncased`, `bert-base-multilingual-cased`, `bert-base-chinese`, `bert-base-german-cased`, `bert-large-uncased-whole-word-masking`, `bert-large-cased-whole-word-masking`, `bert-large-uncased-whole-word-masking-finetuned-squad`, `bert-large-cased-whole-word-masking-finetuned-squad`, `bert-base-cased-finetuned-mrpc`, `bert-base-german-dbmdz-cased`, `bert-base-german-dbmdz-uncased`, `bert-base-japanese`, `bert-base-japanese-whole-word-masking`, `bert-base-japanese-char`, `bert-base-japanese-char-whole-word-masking`, `bert-base-finnish-cased-v1`, `bert-base-finnish-uncased-v1`, `xlnet-base-cased`, `xlnet-large-cased`, `xlm-mlm-en-2048`, `xlm-mlm-ende-1024`, `xlm-mlm-enfr-1024`, `xlm-mlm-enro-1024`, `xlm-mlm-tlm-xnli15-1024`, `xlm-mlm-xnli15-1024`, `xlm-clm-enfr-1024`, `xlm-clm-ende-1024`, `xlm-mlm-17-1280`, `xlm-mlm-100-1280`, `roberta-base`, `roberta-large`, `roberta-large-mnli`, `distilroberta-base`, `roberta-base-openai-detector`, `roberta-large-openai-detector`, `distilbert-base-uncased`, `distilbert-base-uncased-distilled-squad, distilbert-base-german-cased`, `distilbert-base-multilingual-cased`, `albert-base-v1`, `albert-large-v1, albert-xlarge-v1`, `albert-xxlarge-v1`, `albert-base-v2, albert-large-v2`, `albert-xlarge-v2`, `albert-xxlarge-v2, xlm-roberta-base`, `xlm-roberta-large`, `xlm-roberta-large-finetuned-conll02-dutch`, `xlm-roberta-large-finetuned-conll02-spanish`, `xlm-roberta-large-finetuned-conll03-english`, `xlm-roberta-large-finetuned-conll03-german` |

# Download pretrained models

Pretrained models for the sentence retrieval and claim verification steps of the pipeline are available in the [release page][release].

Alternatively they can be downloaded using the provided CLI as follows:

```bash
bash scripts/release.sh --download model "MODEL_NAME"
```
Where MODEL_NAME can be one of:
`ukp-athene+albert-base-v2+albert-base-v2`,
`ukp-athene+bert-base-cased+bert-base-cased`,
`ukp-athene+roberta-base+roberta-base`,
`ukp-athene+xlm-roberta-base+xlm-roberta-base`

## Authors
- **Simone Primarosa** - *Github* ([@simonepri][github:simonepri]) • *Twitter* ([@simoneprimarosa][twitter:simoneprimarosa])

## License
This project is licensed under the MIT License - see the [license][license] file for details.
Some of the files are licensed with the BSD or the Apache-2.0 license.
Please refer to the header of the files for more.

<!-- Links -->
[license]: https://github.com/simonepri/fever-transformers/tree/master/license
[release]: https://github.com/simonepri/fever-transformers/releases

[github:simonepri]: https://github.com/simonepri
[twitter:simoneprimarosa]: http://twitter.com/intent/user?screen_name=simoneprimarosa

[run:colab]: https://colab.research.google.com/drive/1hhJL-VQ__Qh_HsDb6WvflTlNJnEXTlR9

[link:fever]: http://fever.ai
