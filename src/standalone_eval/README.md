Evaluation and Submission
==================

### Task Definition
Given an audio and a natural language query, our task requires a system to retrieve the most relevant moments in the audio.

### Evaluation
At project root, run
```
python src/evaluate.py --config config.yml --model_path results/best_checkpoint.pth
```
This command will use [evaluate.py](src/evaluate.py) to evaluate the provided prediction file [sample_val_preds.jsonl](sample_val_preds.jsonl), the output will be written into `sample_val_preds_metrics.json`. 

### Submission
```
python src/create_submission.py --config config.yml --model_path results/best_checkpoint.pth
```
This command creates `private_submission.json`. Submit this file to us.

### Format
The prediction file [sample_val_preds.jsonl](sample_val_preds.jsonl) is in [JSON Line](https://jsonlines.org/) format, each row of the files can be loaded as a single `dict` in Python.
Below is an example of a single line in the prediction file:
```
{
  "qid": 2579,
  "query": "A girl and her mother cooked while talking with each other on facetime.",
  "vid": "NUsG9BgSes0_210.0_360.0",
  "pred_relevant_windows": [
    [0, 70],
    [78, 146],
    [0, 146],
    ...
  ],  
}
```

| entry | description |
| --- | ----|
| `qid` | `int`, unique query id |
| `query` | `str`, natural language query, not used by the evaluation script | 
| `vid` | `str`, unique audio id (vid is originally video id used in video moment retrieval) | 
| `pred_relevant_windows` | `list(list)`, moment retrieval predictions. Each sublist contains two elements, `[start (seconds), end (seconds)]`| 
