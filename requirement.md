### Objective:

I need to complete a work in project in my company. The task is to Given non-standard forms, perform key information extraction task using open source VLM.
Candidates. 2 sets of models. 1 is general VLM's and second is document specific VLM.

Input must be scanned document available online like non standard invoice.

### Candidates

#### Document specific

* GLM-OCR
* Paddle VL-1.5
* PPchatOCRV4

#### General VLM's

* Qwen VLM'S from smallest of 0.5/0.8b upto 7b

**Qwen3.5 Ollama model candidates (all fit in 4GB VRAM):**

| Tag           | Disk  | VRAM  | Context |
|---------------|-------|-------|---------|
| qwen3.5:0.8b  | 1.0GB | <2GB  | 256K    |
| qwen3.5:2b    | 2.7GB | ~3GB  | 256K    |
| qwen3.5:4b    | 3.4GB | ~4GB  | 256K    |


### End goal

Once I am happy with model results in this R and D. I have to be easily replacate this in company server without much hassle.
