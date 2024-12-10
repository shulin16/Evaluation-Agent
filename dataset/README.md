# Open-Ended User Query Dataset

We compiled the final open-ended user query dataset into the `dataset/open_ended_user_questions.json` file

### 1. Calculate statistics:
By running the command, you can view the dataset’s statistics across different categories.

```bash
python dataset/calculate_statistics.py
```


### 2. View the list of questions for each tag:
By running this command, you can obtain the question list corresponding to each tag.

```bash
python dataset/reorganize_data.py
```
will output json file to `dataset/open_ended_user_questions_summary.json`

