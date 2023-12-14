# openfabric-test
This is project is made for answering scientific questions.

To install the required packages, run:

```bash
pip install -r requirements.txt
```

## How to run the code
As soon as you install the required packages at the `chating.py` file you will find the following list of questions:
```
data = {"text": ["What is atomic number", "What is the capital of France", "What is DNA"]}
```
feel free to change it as much as you need to add your desired questions

Run the ```start.sh``` file
```
sh start.sh
```
Then run the `chating.py` file
```
python chating.py
```

## Methodology
- The model answers the question based on two steps:
  - Retrieving a specifi context using Dense Passage Retrival (DPR)
      - Using the facebook/rag-token-nq
      - A dummy dump for wikipedia
      - and a bert based uncased model as quetion encoder
  - Using a Flan-t5 to answer the question conditioned by the retrieved context
  
